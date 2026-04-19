# Copyright 2026 - Training loss functions for DISCO
# Reproduces Section A.5 of the paper:
#   A.5.1 Structure diffusion losses (MSE + Smooth LDDT)
#   A.5.2 Sequence diffusion losses (Masked diffusion cross-entropy)
#   A.5.3 Full training loss (Eq. S23)
#
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn.functional as F

from disco.data.constants import MASK_TOKEN_IDX


# ---------------------------------------------------------------------------
# A.5.1  Structure Diffusion Losses
# ---------------------------------------------------------------------------


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def weighted_rigid_align(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted rigid alignment of predicted to ground truth coordinates (Eq. S19).

    Computes the optimal rotation and translation that minimizes the weighted
    RMSD between predicted and ground truth coordinates, using Kabsch algorithm.

    Args:
        pred_coords: Predicted coordinates [..., N_atom, 3].
        gt_coords: Ground truth coordinates [..., N_atom, 3].
        weights: Per-atom weights [..., N_atom].
        mask: Optional atom mask [..., N_atom]. 1 = valid, 0 = ignore.

    Returns:
        Aligned ground truth coordinates [..., N_atom, 3].
    """
    # Cast all inputs to float32 for numerical stability (SVD/det need float32)
    pred_coords = pred_coords.float()
    gt_coords = gt_coords.float()
    weights = weights.float()
    if mask is not None:
        mask = mask.float()

    if mask is not None:
        weights = weights * mask

    # Normalize weights
    w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    w_norm = weights / w_sum  # [..., N_atom]

    # Weighted centroids
    pred_center = (w_norm.unsqueeze(-1) * pred_coords).sum(dim=-2, keepdim=True)
    gt_center = (w_norm.unsqueeze(-1) * gt_coords).sum(dim=-2, keepdim=True)

    pred_centered = pred_coords - pred_center
    gt_centered = gt_coords - gt_center

    # Weighted cross-covariance matrix  [..., 3, 3]
    H = torch.einsum(
        "...i,...ij,...ik->...jk",
        w_norm,
        pred_centered,
        gt_centered,
    )

    # SVD for optimal rotation (cast to float32 - SVD doesn't support bf16)
    H_f32 = H.float()
    U, S, Vh = torch.linalg.svd(H_f32)
    U, Vh = U.to(H.dtype), Vh.to(H.dtype)

    # Handle reflections (compute in float32)
    R_candidate = torch.bmm(
        Vh.reshape(-1, 3, 3).float().transpose(-1, -2),
        U.reshape(-1, 3, 3).float().transpose(-1, -2),
    )
    d = torch.det(R_candidate).reshape(H.shape[:-2])
    sign = torch.ones_like(d)
    sign[d < 0] = -1.0

    # Correct for reflection
    Vh_corrected = Vh.clone()
    Vh_corrected[..., -1, :] = Vh[..., -1, :] * sign.unsqueeze(-1)

    # Rotation matrix R = V @ U^T
    R = torch.einsum("...ij,...kj->...ik", Vh_corrected, U)

    # Apply rotation and translation to get aligned GT
    gt_aligned = torch.einsum("...ij,...nj->...ni", R, gt_centered) + pred_center

    return gt_aligned


def mse_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    is_dna: torch.Tensor | None = None,
    is_rna: torch.Tensor | None = None,
    is_ligand: torch.Tensor | None = None,
    a_dna: float = 5.0,
    a_rna: float = 5.0,
    a_ligand: float = 10.0,
) -> torch.Tensor:
    """Structure MSE loss with weighted rigid alignment (Eq. S19-S21).

    Args:
        pred_coords: Predicted denoised structure [..., N_atom, 3].
        gt_coords: Ground truth structure [..., N_atom, 3].
        atom_mask: Valid atom mask [..., N_atom].
        is_dna: Per-atom DNA flag [..., N_atom].
        is_rna: Per-atom RNA flag [..., N_atom].
        is_ligand: Per-atom ligand flag [..., N_atom].
        a_dna, a_rna, a_ligand: Up-weighting factors for non-protein atoms.

    Returns:
        Scalar MSE loss (Eq. S21).
    """
    # Cast to float32 for numerical stability (SVD/det don't support bf16)
    pred_coords = pred_coords.float()
    gt_coords = gt_coords.float()
    atom_mask = atom_mask.float()

    # Compute per-atom weights (Eq. S20)
    weights = torch.ones_like(atom_mask, dtype=pred_coords.dtype)
    if is_dna is not None:
        weights = weights + is_dna.float() * a_dna
    if is_rna is not None:
        weights = weights + is_rna.float() * a_rna
    if is_ligand is not None:
        weights = weights + is_ligand.float() * a_ligand

    # Rigid alignment (Eq. S19)
    gt_aligned = weighted_rigid_align(pred_coords, gt_coords, weights, atom_mask)

    # MSE (Eq. S21): L_MSE = 1/3 * mean_l( w_l * ||x_l^struct - x_l^GT-aligned||^2 )
    sq_diff = ((pred_coords - gt_aligned) ** 2).sum(dim=-1)  # [..., N_atom]
    weighted_sq_diff = weights * sq_diff * atom_mask

    n_valid = atom_mask.sum(dim=-1).clamp(min=1)
    loss = (weighted_sq_diff.sum(dim=-1) / n_valid) / 3.0

    return loss.mean()


def smooth_lddt_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    is_dna: torch.Tensor | None = None,
    is_rna: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth LDDT loss (Algorithm 6 from the paper).

    A differentiable approximation of the LDDT score using sigmoid functions
    instead of hard thresholds.

    Args:
        pred_coords: Predicted coordinates [..., N_atom, 3].
        gt_coords: Ground truth coordinates [..., N_atom, 3].
        atom_mask: Valid atom mask [..., N_atom].
        is_dna: Per-atom DNA indicator [..., N_atom].
        is_rna: Per-atom RNA indicator [..., N_atom].

    Returns:
        Scalar smooth LDDT loss (1 - lddt).
    """
    # Pairwise distances
    # pred: [..., N, 3] -> [..., N, N]
    pred_dists = torch.cdist(pred_coords, pred_coords)  # [..., N, N]
    gt_dists = torch.cdist(gt_coords, gt_coords)  # [..., N, N]

    # Distance difference (Algorithm 6, line 6)
    delta = torch.abs(gt_dists - pred_dists)  # [..., N, N]

    # Smooth LDDT score (Algorithm 6, line 7)
    # epsilon = 1/4 * [sigmoid(0.5 - delta) + sigmoid(1 - delta) + sigmoid(2 - delta) + sigmoid(4 - delta)]
    eps_lm = 0.25 * (
        torch.sigmoid(0.5 - delta)
        + torch.sigmoid(1.0 - delta)
        + torch.sigmoid(2.0 - delta)
        + torch.sigmoid(4.0 - delta)
    )

    # Determine nucleotide mask for inclusion radius
    is_nucleotide = torch.zeros_like(atom_mask)
    if is_dna is not None:
        is_nucleotide = is_nucleotide + is_dna.float()
    if is_rna is not None:
        is_nucleotide = is_nucleotide + is_rna.float()
    is_nucleotide = is_nucleotide.clamp(max=1.0)

    # Inclusion radius: 30A for nucleotides, 15A for others (Algorithm 6, line 10)
    # c_lm = (gt_dist < 30) * is_nucleotide + (gt_dist < 15) * (1 - is_nucleotide)
    nuc_l = is_nucleotide.unsqueeze(-1)  # [..., N, 1]
    nuc_m = is_nucleotide.unsqueeze(-2)  # [..., 1, N]
    is_nuc_pair = (nuc_l + nuc_m).clamp(max=1.0)  # [..., N, N]

    inclusion_mask = (
        (gt_dists < 30.0) * is_nuc_pair + (gt_dists < 15.0) * (1.0 - is_nuc_pair)
    )

    # Pair mask: both atoms must be valid
    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)  # [..., N, N]

    # Exclude self-interactions
    eye = torch.eye(pred_coords.shape[-2], device=pred_coords.device)
    non_self = 1.0 - eye
    if pair_mask.ndim > 2:
        non_self = non_self.unsqueeze(0).expand_as(pair_mask)

    c_lm = inclusion_mask * pair_mask * non_self

    # LDDT = mean(c * eps) / mean(c) (Algorithm 6, line 12)
    numerator = (c_lm * eps_lm).sum(dim=-1)  # [..., N]
    denominator = c_lm.sum(dim=-1).clamp(min=1e-8)  # [..., N]

    per_atom_lddt = numerator / denominator
    # Mask and average
    lddt = (per_atom_lddt * atom_mask).sum(dim=-1) / atom_mask.sum(dim=-1).clamp(min=1)

    # Loss = 1 - lddt (Algorithm 6, line 13)
    return (1.0 - lddt).mean()


def structure_diffusion_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    t_hat: torch.Tensor,
    sigma_data: float = 16.0,
    is_dna: torch.Tensor | None = None,
    is_rna: torch.Tensor | None = None,
    is_ligand: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full structure diffusion loss (Eq. S22).

    L_struct = (t_hat^2 + sigma_data^2) / (t_hat + sigma_data)^2 * L_MSE + L_smooth_lddt

    Args:
        pred_coords: Predicted denoised coords [..., N_atom, 3].
        gt_coords: Ground truth coords [..., N_atom, 3].
        atom_mask: Valid atom mask [..., N_atom].
        t_hat: Sampled structure noise level [batch].
        sigma_data: Data normalization constant (16.0).
        is_dna, is_rna, is_ligand: Per-atom type flags.

    Returns:
        Tuple of (weighted_mse_loss, smooth_lddt_loss).
    """
    l_mse = mse_loss(pred_coords, gt_coords, atom_mask, is_dna, is_rna, is_ligand)
    l_lddt = smooth_lddt_loss(pred_coords, gt_coords, atom_mask, is_dna, is_rna)

    # Noise weighting (Eq. S22)
    weight = (t_hat**2 + sigma_data**2) / (t_hat + sigma_data) ** 2
    weighted_mse = weight.mean() * l_mse

    return weighted_mse, l_lddt


# ---------------------------------------------------------------------------
# A.5.2  Sequence Diffusion Losses
# ---------------------------------------------------------------------------


def sequence_diffusion_loss(
    seq_logits: torch.Tensor,
    gt_seq: torch.Tensor,
    mask_rate: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked diffusion cross-entropy loss for sequences (Eq. S8).

    Uses the SUBS parameterization with linear schedule alpha_r = 1-r.
    The loss weight is d(alpha_r)/dr * 1/(1-alpha_r) = -1/r for linear schedule.
    Only computes loss on masked (unknown) positions.

    Args:
        seq_logits: Predicted logits [..., N_token, V] (V = vocab size).
        gt_seq: Ground truth token indices [..., N_token].
        mask_rate: Fraction of tokens that are masked (= 1-r) [...].
        seq_mask: Valid token mask [..., N_token]. 1 = compute loss, 0 = skip.

    Returns:
        Scalar sequence diffusion loss.
    """
    # Cross-entropy loss per token (no reduction)
    V = seq_logits.shape[-1]
    ce = F.cross_entropy(
        seq_logits.reshape(-1, V),
        gt_seq.reshape(-1),
        reduction="none",
    ).reshape(gt_seq.shape)

    # Only compute loss on masked positions (SUBS: only predict at mask positions)
    # mask_rate = 1 - alpha_r  (fraction masked)
    # Loss weight from Eq. S8: d(alpha_r)/dr * 1/(1-alpha_r) = -1/r
    # Since r = 1 - mask_rate, weight = 1/(1 - mask_rate), but this is
    # already accounted for by the uniform sampling of r.
    # In practice (MDLM/SUBS), we compute: -1/r * CE on masked positions
    # r = 1 - mask_rate
    r = 1.0 - mask_rate  # sequence time
    # Avoid division by zero
    weight = 1.0 / r.clamp(min=1e-4)

    # Apply mask
    masked_ce = ce * seq_mask
    n_valid = seq_mask.sum(dim=-1).clamp(min=1)
    per_sample_loss = masked_ce.sum(dim=-1) / n_valid

    # Weight by noise schedule
    if weight.ndim == 0:
        weighted_loss = weight * per_sample_loss
    else:
        weighted_loss = weight * per_sample_loss

    return weighted_loss.mean()


# ---------------------------------------------------------------------------
# A.5.3  Full Training Loss (Eq. S23) + Distogram
# ---------------------------------------------------------------------------


def distogram_loss(
    distogram_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    token_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    no_bins: int = 64,
) -> torch.Tensor:
    """Distogram cross-entropy loss (AlphaFold 2 style).

    Computes pairwise distances between representative atoms (CA for protein),
    bins them, and computes cross-entropy against the predicted distogram.

    Args:
        distogram_logits: Predicted distogram [..., N_token, N_token, no_bins].
        gt_coords: Ground truth atom coordinates [..., N_atom, 3].
        token_mask: Valid token mask [..., N_token].
        atom_to_token_idx: Mapping from atoms to tokens [..., N_atom].
        min_dist: Minimum distance for binning.
        max_dist: Maximum distance for binning.
        no_bins: Number of distance bins.

    Returns:
        Scalar distogram loss.
    """
    N_token = distogram_logits.shape[-2]
    device = distogram_logits.device

    # Get representative atom positions (first atom per token = CA typically)
    # We use a simple approach: for each token, find the first atom
    rep_coords = torch.zeros(
        *gt_coords.shape[:-2], N_token, 3, device=device, dtype=gt_coords.dtype
    )
    rep_mask = torch.zeros(*gt_coords.shape[:-2], N_token, device=device)

    for t in range(N_token):
        atom_idx = (atom_to_token_idx == t).nonzero(as_tuple=True)
        if len(atom_idx[-1]) > 0:
            first_atom = atom_idx[-1][0]
            if gt_coords.ndim == 2:
                rep_coords[t] = gt_coords[first_atom]
            else:
                rep_coords[..., t, :] = gt_coords[..., first_atom, :]
            if rep_mask.ndim == 1:
                rep_mask[t] = 1.0
            else:
                rep_mask[..., t] = 1.0

    # Pairwise distances between representative atoms
    dists = torch.cdist(rep_coords, rep_coords)  # [..., N_token, N_token]

    # Bin edges
    bin_edges = torch.linspace(min_dist, max_dist, no_bins - 1, device=device)

    # Create target bins
    target_bins = torch.bucketize(dists, bin_edges)  # [..., N_token, N_token]

    # Pair mask
    pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    eye = torch.eye(N_token, device=device)
    if pair_mask.ndim > 2:
        eye = eye.unsqueeze(0).expand_as(pair_mask)
    pair_mask = pair_mask * (1.0 - eye) * rep_mask.unsqueeze(-1) * rep_mask.unsqueeze(-2)

    # Cross-entropy
    ce = F.cross_entropy(
        distogram_logits.reshape(-1, no_bins),
        target_bins.reshape(-1),
        reduction="none",
    ).reshape(target_bins.shape)

    masked_ce = ce * pair_mask
    n_valid = pair_mask.sum().clamp(min=1)

    return masked_ce.sum() / n_valid


def full_training_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    t_hat: torch.Tensor,
    seq_logits: torch.Tensor,
    gt_seq: torch.Tensor,
    mask_rate: torch.Tensor,
    seq_mask: torch.Tensor,
    distogram_logits: torch.Tensor,
    token_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    sigma_data: float = 16.0,
    alpha_seq: float = 1.0,
    alpha_mse: float = 4.0,
    alpha_smooth_lddt: float = 4.0,
    alpha_distogram: float = 0.03,
    is_dna: torch.Tensor | None = None,
    is_rna: torch.Tensor | None = None,
    is_ligand: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Full DISCO training loss (Eq. S23).

    L_total = alpha_seq * L_seq + alpha_MSE * L_MSE
              + alpha_smooth_lddt * L_smooth_lddt
              + alpha_distogram * L_distogram

    Args:
        pred_coords: Predicted denoised structure [..., N_atom, 3].
        gt_coords: Ground truth structure [..., N_atom, 3].
        atom_mask: Valid atom mask [..., N_atom].
        t_hat: Sampled structure noise level [batch].
        seq_logits: Predicted sequence logits [..., N_token, V].
        gt_seq: Ground truth sequence tokens [..., N_token].
        mask_rate: Fraction of sequence tokens masked [...].
        seq_mask: Mask for valid+masked sequence positions [..., N_token].
        distogram_logits: Predicted distogram [..., N_token, N_token, no_bins].
        token_mask: Valid token mask [..., N_token].
        atom_to_token_idx: Atom-to-token index map [..., N_atom].
        sigma_data: Data normalization constant.
        alpha_seq, alpha_mse, alpha_smooth_lddt, alpha_distogram: Loss weights.
        is_dna, is_rna, is_ligand: Per-atom type flags.

    Returns:
        Dict with 'total', 'seq', 'mse', 'smooth_lddt', 'distogram' losses.
    """
    # Structure losses (Eq. S22)
    weighted_mse, l_lddt = structure_diffusion_loss(
        pred_coords, gt_coords, atom_mask, t_hat, sigma_data,
        is_dna, is_rna, is_ligand,
    )

    # Sequence loss (Eq. S8)
    l_seq = sequence_diffusion_loss(seq_logits, gt_seq, mask_rate, seq_mask)

    # Distogram loss
    l_distogram = distogram_loss(
        distogram_logits, gt_coords, token_mask, atom_to_token_idx,
    )

    # Full loss (Eq. S23)
    total = (
        alpha_seq * l_seq
        + alpha_mse * weighted_mse
        + alpha_smooth_lddt * l_lddt
        + alpha_distogram * l_distogram
    )

    return {
        "total": total,
        "seq": l_seq.detach(),
        "mse": weighted_mse.detach(),
        "smooth_lddt": l_lddt.detach(),
        "distogram": l_distogram.detach(),
    }
