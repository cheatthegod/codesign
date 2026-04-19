# Copyright 2026 - Training script for DISCO
# Reproduces the training procedure described in Section A.5 of the paper:
#   "General Multimodal Protein Design Enables DNA-Encoding of Chemistry"
#
# Key training details (Table S1):
#   - 32 L40S GPUs, 11 days, 160,000 steps
#   - Adam optimizer: lr=0.00018, beta=(0.9, 0.95), weight_decay=1e-8
#   - Linear warmup 1000 steps, decay 0.95 every 50,000 steps
#   - EMA decay 0.999, gradient clipping 10.0
#   - Loss: alpha_seq*L_seq + alpha_MSE*L_MSE + alpha_lddt*L_lddt + alpha_disto*L_disto
#   - Noise: structure sigma=sigma_data*exp(-1.2+1.5*N(0,1)), sequence r~U(0,1)
#
# Licensed under the Apache License, Version 2.0

import copy
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from disco.data.constants import MASK_TOKEN_IDX
from disco.data.train_data_pipeline import (
    get_training_dataloader,
    noise_sequence,
    noise_structure,
)
from disco.data.prot2text_adapter import get_prot2text_dataloader
from disco.data.pdb_complex_adapter import get_pdb_complex_dataloader, get_cached_pdb_complex_dataloader
from disco.model.disco import DISCO
from disco.model.losses import full_training_loss
from disco.model.utils import centre_random_augmentation
from disco.utils.seed import seed_everything
from disco.utils.torch_utils import to_device
from runner.utils import print_config_tree

OmegaConf.register_new_resolver("gt", lambda a, b: a > b, replace=True)
rootutils.setup_root(__file__, indicator=".project-root")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) for model parameters
# ---------------------------------------------------------------------------


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated each step
    via: shadow = decay * shadow + (1 - decay) * param

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (0.999 in paper Table S1).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model parameters with EMA shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)


# ---------------------------------------------------------------------------
# Learning Rate Scheduler (Table S1)
# ---------------------------------------------------------------------------


def get_lr(step: int, configs) -> float:
    """Compute learning rate with linear warmup and step decay.

    Schedule from Table S1:
    - Linear warmup over warmup_steps (1000)
    - Step decay: multiply by decay_factor (0.95) every decay_every_n_steps (50000)

    Args:
        step: Current training step.
        configs: Training config with scheduler parameters.

    Returns:
        Current learning rate.
    """
    base_lr = configs.optimizer.lr
    warmup_steps = configs.scheduler.warmup_steps
    decay_factor = configs.scheduler.decay_factor
    decay_every = configs.scheduler.decay_every_n_steps

    # Linear warmup
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    # Step decay
    n_decays = (step - warmup_steps) // decay_every
    return base_lr * (decay_factor ** n_decays)


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------


def training_step(
    model: DISCO,
    batch: dict,
    configs: DictConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Execute one training step: forward pass + loss computation.

    This implements the training procedure from Section A.5:
    1. Extract ground truth coordinates and sequence from the batch
    2. Sample structure noise t_hat and sequence noise r independently
    3. Create noised inputs for both modalities
    4. Run the model trunk (get_pairformer_output) with cross-modal recycling
    5. Run the diffusion module to predict denoised x0
    6. Compute the full training loss (Eq. S23)

    Args:
        model: DISCO model.
        batch: Training batch dictionary.
        configs: Training configuration.
        device: Target device.

    Returns:
        Dictionary of loss values.
    """
    input_feature_dict = batch["input_feature_dict"]

    # Move ALL tensor features to device (needed for diffusion module scatter ops)
    for key in list(input_feature_dict.keys()):
        v = input_feature_dict[key]
        if torch.is_tensor(v):
            input_feature_dict[key] = v.to(device)
        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
            pass  # Keep non-tensor values as-is
        # AtomArray and TokenArray objects stay on CPU

    # --- Ground truth ---
    # GT coordinates: from the atom positions in the feature dict
    # The ref_pos contains the original reference positions
    gt_coords = input_feature_dict["ref_pos"].float().to(device)

    # GT sequence: use masked_prot_restype as the reference length
    # After cropping, masked_prot_restype has the correct protein token count
    masked_prot = input_feature_dict["masked_prot_restype"]
    n_prot = masked_prot.shape[0]

    gt_seq = input_feature_dict.get("gt_seq")
    if gt_seq is None or gt_seq.shape[0] != n_prot:
        # Fallback: use UNK for gt_seq if length mismatch (after crop)
        gt_seq = torch.full((n_prot,), PRO_STD_RESIDUES.get("UNK", 20), dtype=torch.long)
    gt_seq = gt_seq.to(device)

    # Atom mask: which atoms are resolved
    atom_mask = input_feature_dict.get("ref_mask", torch.ones(gt_coords.shape[:-1]))
    atom_mask = atom_mask.float().to(device)

    # Token-level masks
    prot_res_mask = input_feature_dict.get("prot_residue_mask")
    if prot_res_mask is not None:
        prot_res_mask = prot_res_mask.float().to(device)

    # --- Sample noise for structure (Section A.5.1) ---
    sigma_data = configs.noise.sigma_data
    noised_coords, t_hat = noise_structure(
        gt_coords,
        sigma_data=sigma_data,
        log_mean=configs.noise.struct_log_mean,
        log_std=configs.noise.struct_log_std,
    )

    # Apply SE(3) augmentation (paper: "We inject SE(3) symmetries softly through
    # data augmentation rather than architectural constraints")
    if configs.use_se3_augmentation:
        noised_coords = centre_random_augmentation(
            noised_coords.unsqueeze(0), N_sample=1, s_trans=1.0
        ).squeeze(0)

    # --- Sample noise for sequence (Section A.5.2) ---
    noised_seq, mask_rate, masked_positions = noise_sequence(
        gt_seq,
        mask_token_idx=MASK_TOKEN_IDX,
        noise_min=configs.noise.get("seq_noise_min", 0.0),
        noise_max=configs.noise.get("seq_noise_max", 0.95),
    )

    # Update input features with noised data (must match original dtype - long)
    input_feature_dict["masked_prot_restype"] = noised_seq.long().to(device)

    # Mask reference information for masked positions to prevent leakage
    # (Section A.5.2: "we ensure that all amino acid residue information is not
    # leaked to the model by masking all reference information features")
    # This is handled by the TaskManager during data loading, but we also need
    # to update the features for dynamically masked positions.

    # --- Forward pass through trunk and diffusion ---
    # Use autocast for model forward passes only
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        s_inputs, s_trunk, z_trunk, s_orig, z_orig, lm_logits, encoding_dict = (
            model.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=model.N_cycle,
                task_info={},
                sigma_seq=None,
                xt_noised_struct=None,
                sigma=None,
            )
        )

        sigma_tensor = t_hat.view(1)
        sigma_seq_tensor = mask_rate.view(1)

        x0_struct, seq_logits_pre = model.diffusion_module(
            x_noisy=noised_coords.unsqueeze(0),
            t_hat_noise_level_struct=sigma_tensor,
            t_hat_noise_level_seq=sigma_seq_tensor,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk.unsqueeze(0),
            z_trunk=z_trunk.unsqueeze(0),
            s_skip=None,
            z_skip=None,
        )

    # Squeeze N_sample=1 dimension
    x0_struct = x0_struct.squeeze(0)  # [N_atom, 3]
    seq_logits_pre = seq_logits_pre.squeeze(0) if seq_logits_pre.ndim > 2 else seq_logits_pre

    # Cast outputs to float32 for loss computation
    x0_struct = x0_struct.float()
    seq_logits_pre = seq_logits_pre.float()

    # Apply SUBS parameterization to sequence logits
    # (Section A.5.2: mask token logit -> -inf)
    seq_logits = model.apply_subs_parameterization(
        seq_logits_pre.clone(),
        noised_seq,
        exclude_unk=False,
        enforce_unmask_stay=False,  # During training, don't force unmask stay
    )

    # --- Distogram prediction ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        distogram_logits = model.distogram_head(z_trunk)
    distogram_logits = distogram_logits.float()

    # --- Compute full training loss (Eq. S23) ---
    # Sequence mask: only compute loss on masked (unknown) positions
    seq_loss_mask = masked_positions.float().to(device)

    # Occupancy filtering: mask out unreliable residues
    # Only apply if dimensions match (may mismatch after cropping)
    res_occ_mask = input_feature_dict.get("res_occ_cutoff_mask")
    if res_occ_mask is not None and res_occ_mask.numel() == seq_loss_mask.numel():
        seq_loss_mask = seq_loss_mask * res_occ_mask.float().to(device)

    # Histag mask
    histag_mask = input_feature_dict.get("histag_mask")
    if histag_mask is not None and histag_mask.numel() == seq_loss_mask.numel():
        seq_loss_mask = seq_loss_mask * (1.0 - histag_mask.float().to(device))

    # Type flags for structure loss weighting
    is_dna = input_feature_dict.get("is_dna")
    is_rna = input_feature_dict.get("is_rna")
    is_ligand = input_feature_dict.get("is_ligand")
    if is_dna is not None:
        is_dna = is_dna.float().to(device)
    if is_rna is not None:
        is_rna = is_rna.float().to(device)
    if is_ligand is not None:
        is_ligand = is_ligand.float().to(device)

    # Token mask for distogram
    token_mask = torch.ones(
        distogram_logits.shape[-2], device=device
    )

    atom_to_token_idx = input_feature_dict.get("atom_to_token_idx")
    if atom_to_token_idx is not None:
        atom_to_token_idx = atom_to_token_idx.to(device)

    loss_dict = full_training_loss(
        pred_coords=x0_struct,
        gt_coords=gt_coords,
        atom_mask=atom_mask,
        t_hat=sigma_tensor,
        seq_logits=seq_logits,
        gt_seq=gt_seq,
        mask_rate=mask_rate.to(device),
        seq_mask=seq_loss_mask,
        distogram_logits=distogram_logits.squeeze(0) if distogram_logits.ndim > 3 else distogram_logits,
        token_mask=token_mask,
        atom_to_token_idx=atom_to_token_idx,
        sigma_data=sigma_data,
        alpha_seq=configs.loss_weights.seq,
        alpha_mse=configs.loss_weights.mse,
        alpha_smooth_lddt=configs.loss_weights.smooth_lddt,
        alpha_distogram=configs.loss_weights.distogram,
        is_dna=is_dna,
        is_rna=is_rna,
        is_ligand=is_ligand,
    )

    return loss_dict


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Main training class for DISCO.

    Handles:
    - Environment setup (Fabric, distributed training)
    - Model initialization and optional checkpoint loading
    - Training loop with gradient accumulation
    - EMA parameter tracking
    - Checkpointing and logging

    Args:
        configs: Hydra DictConfig with all training parameters.
    """

    def __init__(self, configs: DictConfig) -> None:
        self.configs = configs
        self.global_step = 0
        self.best_loss = float("inf")

        self.init_env()
        self.init_model()
        self.init_optimizer()
        self.init_ema()

        if configs.load_checkpoint_path is not None:
            self.load_checkpoint(configs.load_checkpoint_path)

    def init_env(self):
        """Initialize distributed environment with Lightning Fabric."""
        n_gpus = torch.cuda.device_count()
        use_ddp = n_gpus > 1 and self.configs.get("use_ddp", True)

        if use_ddp:
            # find_unused_parameters=True required because not all params
            # are used every step (e.g., cross-modal recycling is disabled)
            # static_graph=True avoids the "mark variable ready" error
            strategy = DDPStrategy(
                find_unused_parameters=True,
                static_graph=True,
            )
        else:
            strategy = "auto"

        self.fabric = Fabric(
            strategy=strategy,
            num_nodes=self.configs.fabric.num_nodes,
            precision=self.configs.dtype,
        )
        self.fabric.launch()
        self.device = self.fabric.device
        torch.cuda.set_device(self.device)

        # CUDA optimizations
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0,8.9,9.0"
        torch.set_float32_matmul_precision("high")
        if self.configs.get("use_deepspeed_evo_attention", False):
            env = os.getenv("CUTLASS_PATH", None)
            if env is not None:
                self.print("DeepSpeed EvoformerAttention enabled")

        self.print(
            f"Fabric initialized: rank={self.fabric.global_rank}, "
            f"world_size={self.fabric.world_size}"
        )

    def init_model(self):
        """Initialize the DISCO model with structure encoder."""
        structure_encoder = None
        if self.configs.structure_encoder.use_structure_encoder:
            structure_encoder = hydra.utils.instantiate(
                self.configs.structure_encoder.args
            )

        sequence_sampling_strategy = hydra.utils.instantiate(
            self.configs.sequence_sampling_strategy
        )

        self.model = DISCO(
            self.configs, structure_encoder, sequence_sampling_strategy
        ).to(self.device)

        # Freeze the pre-trained language model (Section A.5: "we fully freeze
        # the pre-trained DPLM 650M weights during training")
        if self.configs.get("freeze_lm", True):
            for param in self.model.lm_module.parameters():
                param.requires_grad = False
            self.print("Frozen pLM (DPLM 650M) weights")

        # Enable gradient checkpointing on pairformer to save memory
        # This allows training on much longer proteins (384+ tokens)
        if self.configs.get("gradient_checkpointing", True):
            from torch.utils.checkpoint import checkpoint as torch_ckpt
            orig_pairformer = self.model.pairformer_stack.forward
            def ckpt_pairformer(s, z, **kwargs):
                def run(s, z):
                    return orig_pairformer(s, z, **kwargs)
                return torch_ckpt(run, s, z, use_reentrant=False)
            self.model.pairformer_stack.forward = ckpt_pairformer
            self.print("Enabled gradient checkpointing on PairformerStack")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.print(
            f"Model: {total_params / 1e6:.1f}M total, "
            f"{trainable_params / 1e6:.1f}M trainable"
        )

    def init_optimizer(self):
        """Initialize Adam optimizer and set up Fabric wrapping."""
        opt_cfg = self.configs.optimizer
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
            eps=opt_cfg.eps,
        )

        # Wrap model and optimizer with Fabric for distributed training
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Mark internal methods as forward methods so Fabric allows calling them
        self.model.mark_forward_method("get_pairformer_output")
        self.model.mark_forward_method("predict_x0_seq_struct")
        self.model.mark_forward_method("apply_subs_parameterization")

    def init_ema(self):
        """Initialize EMA tracker."""
        self.ema = EMA(self.model, decay=self.configs.ema_decay)
        self.print(f"EMA initialized with decay={self.configs.ema_decay}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model, optimizer, and EMA state from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            self.print(f"Checkpoint not found: {checkpoint_path}")
            return

        self.print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            # Handle DDP prefix
            sample_key = next(iter(state_dict))
            if sample_key.startswith("module."):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

            # Filter mismatched shapes
            current = self.model.state_dict()
            filtered = OrderedDict()
            for k, v in state_dict.items():
                if k in current and v.shape == current[k].shape:
                    filtered[k] = v
                else:
                    self.print(f"Skipping '{k}': shape mismatch")
            self.model.load_state_dict(filtered, strict=False)

        # Load optimizer state
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                self.print(f"Could not load optimizer state: {e}")

        # Load EMA state
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

        # Load training state
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        self.print(f"Resumed from step {self.global_step}")

    def save_checkpoint(self, suffix: str = ""):
        """Save model, optimizer, and EMA state to checkpoint.

        Args:
            suffix: Optional suffix for the checkpoint filename.
        """
        if not self.fabric.is_global_zero:
            return

        output_dir = Path(self.configs.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"disco_step_{self.global_step}{suffix}.pt"
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "global_step": self.global_step,
            "configs": OmegaConf.to_container(self.configs, resolve=True),
        }

        path = output_dir / filename
        torch.save(checkpoint, path)
        self.print(f"Checkpoint saved: {path}")

        # Also save EMA checkpoint separately (Section A.5.5: "We choose our
        # final checkpoint by evaluating EMA checkpoints")
        self.ema.apply_shadow(self.model)
        ema_checkpoint = {"model": self.model.state_dict()}
        ema_path = output_dir / f"disco_ema_step_{self.global_step}{suffix}.pt"
        torch.save(ema_checkpoint, ema_path)
        self.ema.restore(self.model)
        self.print(f"EMA checkpoint saved: {ema_path}")

    def train(self):
        """Main training loop.

        Implements the full training procedure:
        1. Load training data
        2. For each step:
            a. Get batch, compute forward pass and loss
            b. Backward pass with gradient accumulation
            c. Gradient clipping (norm=10.0)
            d. Optimizer step with LR scheduling
            e. EMA update
            f. Logging and checkpointing
        """
        self.print("=" * 60)
        self.print("Starting DISCO training")
        self.print(f"Max steps: {self.configs.max_steps}")
        self.print(f"Output dir: {self.configs.output_dir}")
        self.print("=" * 60)

        # Create output directory
        os.makedirs(self.configs.output_dir, exist_ok=True)

        # Load training data - three modes:
        # 1. PDB complex data (protein + ligand) for conditional training
        # 2. Prot2Text data (protein only) for unconditional training
        # 3. Generic PDB directory
        if self.configs.get("cache_dir") is not None:
            dataloader = get_cached_pdb_complex_dataloader(self.fabric, self.configs)
            self.print(f"Using cached PDB complex dataset: {len(dataloader.dataset)} samples")
        elif self.configs.get("complex_pdb_dir") is not None:
            dataloader = get_pdb_complex_dataloader(self.fabric, self.configs)
            self.print(f"Using PDB complex dataset: {len(dataloader.dataset)} samples")
        elif self.configs.get("prot2text_csv") is not None:
            dataloader = get_prot2text_dataloader(self.fabric, self.configs)
            self.print(f"Using Prot2Text dataset: {len(dataloader.dataset)} samples")
        else:
            dataloader = get_training_dataloader(self.fabric, self.configs)
            self.print(f"Training dataset: {len(dataloader.dataset)} samples")

        # Training loop
        self.model.train()
        accumulation_steps = self.configs.gradient_accumulation_steps
        running_losses = {}
        step_time_start = time.time()

        while self.global_step < self.configs.max_steps:
            # Epoch loop
            if hasattr(dataloader.sampler, "set_epoch"):
                epoch = self.global_step // len(dataloader)
                dataloader.sampler.set_epoch(epoch)

            for batch in dataloader:
                if batch is None:
                    continue

                if self.global_step >= self.configs.max_steps:
                    break

                # Update learning rate
                lr = get_lr(self.global_step, self.configs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Mixed precision context
                precision_dtype = {
                    "fp32": torch.float32,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }.get(self.configs.dtype, torch.bfloat16)

                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=precision_dtype)
                    if torch.cuda.is_available()
                    else nullcontext()
                )

                # Forward pass (disable autocast - training_step handles precision)
                try:
                    loss_dict = training_step(
                        model=self.model,
                        batch=batch,
                        configs=self.configs,
                        device=self.device,
                    )

                    loss = loss_dict["total"] / accumulation_steps

                    # Backward pass
                    self.fabric.backward(loss)

                except Exception as e:
                    logger.warning(
                        f"Step {self.global_step}: training step failed: {e}"
                    )
                    # Use zero loss to keep DDP ranks in sync (avoid NCCL hang)
                    dummy_loss = sum(p.sum() * 0 for p in self.model.parameters() if p.requires_grad)
                    self.fabric.backward(dummy_loss / accumulation_steps)

                # Gradient accumulation
                if (self.global_step + 1) % accumulation_steps == 0:
                    # Gradient clipping (Table S1: norm=10.0)
                    if self.configs.gradient_clip_norm > 0:
                        self.fabric.clip_gradients(
                            self.model,
                            self.optimizer,
                            max_norm=self.configs.gradient_clip_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # EMA update (Table S1: decay=0.999)
                    self.ema.update(self.model)

                # Accumulate losses for logging
                for key, val in loss_dict.items():
                    if key not in running_losses:
                        running_losses[key] = 0.0
                    running_losses[key] += val.item()

                self.global_step += 1

                # --- Logging ---
                if self.global_step % self.configs.log_every_n_steps == 0:
                    step_time = time.time() - step_time_start
                    n_log = self.configs.log_every_n_steps

                    avg_losses = {
                        k: v / n_log for k, v in running_losses.items()
                    }

                    self.print(
                        f"Step {self.global_step}/{self.configs.max_steps} | "
                        f"lr={lr:.6f} | "
                        f"loss={avg_losses.get('total', 0):.4f} | "
                        f"seq={avg_losses.get('seq', 0):.4f} | "
                        f"mse={avg_losses.get('mse', 0):.4f} | "
                        f"lddt={avg_losses.get('smooth_lddt', 0):.4f} | "
                        f"disto={avg_losses.get('distogram', 0):.4f} | "
                        f"time={step_time:.1f}s"
                    )

                    # Log to Fabric loggers
                    if self.fabric.is_global_zero:
                        for k, v in avg_losses.items():
                            self.fabric.log(f"train/{k}", v, step=self.global_step)
                        self.fabric.log("train/lr", lr, step=self.global_step)

                    running_losses = {}
                    step_time_start = time.time()

                # --- Checkpointing ---
                if self.global_step % self.configs.save_every_n_steps == 0:
                    self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint(suffix="_final")
        self.print("Training complete!")

    def print(self, msg: str):
        """Log message on rank 0 only."""
        if self.fabric.is_global_zero:
            logging.info(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(configs: DictConfig):
    """Main entry point for DISCO training."""
    LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    # Determine log file path
    output_dir = configs.get("output_dir", "./train_output_conditional")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    # Set up root logger with both console and file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with immediate flush (rank 0 only)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if rank == 0:
        fh_stream = open(log_file, "w", buffering=1)  # line-buffered
        file_handler = logging.StreamHandler(fh_stream)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    try:
        print_config_tree(configs, resolve=True)
    except Exception as e:
        logging.warning(f"Could not print config tree: {e}")
        logging.info(f"Training config keys: {list(configs.keys())}")

    # Seed
    seed_everything(seed=configs.seed, deterministic=configs.deterministic)

    # Train
    trainer = Trainer(configs)
    trainer.train()


if __name__ == "__main__":
    main()
