import math

import torch
from torch import linalg as LA


@torch.no_grad()
def _matrix_sqrt_psd(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # symmetric PSD matrix square root via eigen decomposition
    Sigma = 0.5 * (Sigma + Sigma.t())
    d = Sigma.shape[0]
    Sigma = Sigma + eps * \
        torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    evals, evecs = torch.linalg.eigh(Sigma)  # ascending
    evals = torch.clamp(evals, min=0.0).sqrt()
    return (evecs * evals.unsqueeze(0)) @ evecs.t()


@torch.no_grad()
def _randomized_svd_fallback(
    W: torch.Tensor,
    q: int,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust randomized SVD fallback for CUDA SVD convergence failures.

    Computes an approximate rank-q SVD via a randomized range finder and SVD of the
    small matrix B = Q^T W (shape q x n). If CUDA SVD fails, we move B to CPU float64
    and compute the SVD there (cheap + stable).
    """
    m, n = W.shape
    q = int(min(int(q), int(m), int(n)))
    if q <= 0:
        raise ValueError(f"q must be >= 1, got {q}")

    Omega = torch.randn((n, q), device=W.device, dtype=W.dtype)
    Y = W @ Omega
    for _ in range(max(0, int(niter))):
        Y = W @ (W.t() @ Y)

    Q, _ = LA.qr(Y, mode="reduced")  # (m, q)
    B = Q.t() @ W  # (q, n)

    try:
        Ub, S, Vh = LA.svd(B, full_matrices=False)
    except LA.LinAlgError:
        Ub_cpu, S_cpu, Vh_cpu = LA.svd(B.to(device="cpu", dtype=torch.float64), full_matrices=False)
        Ub = Ub_cpu.to(device=W.device, dtype=W.dtype, non_blocking=(W.device.type == "cuda"))
        S = S_cpu.to(device=W.device, dtype=W.dtype, non_blocking=(W.device.type == "cuda"))
        Vh = Vh_cpu.to(device=W.device, dtype=W.dtype, non_blocking=(W.device.type == "cuda"))

    U = Q @ Ub  # (m, q)
    V = Vh.t()  # (n, q)
    return U, S, V


@torch.no_grad()
def _svd_lowrank_safe(
    W: torch.Tensor,
    q: int,
    niter: int = 2,
    max_retries: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Safe wrapper around torch.svd_lowrank.

    - Retries a few times (svd_lowrank is randomized; a new draw may succeed).
    - If it still fails, falls back to a randomized SVD whose SVD stage is done on
      a small matrix, with CPU float64 fallback for numerical robustness.
    """
    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be >= 1, got {q}")

    last_err = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            return torch.svd_lowrank(W, q=q, niter=int(niter))
        except (LA.LinAlgError, RuntimeError) as e:
            last_err = e
            if attempt < max_retries:
                continue

    try:
        print(
            f"[cur_deim_gpu] svd_lowrank failed after retries (q={q}, niter={niter}). "
            f"Falling back to randomized SVD with CPU SVD for B. err={last_err}"
        )
    except Exception:
        pass
    return _randomized_svd_fallback(W, q=q, niter=int(niter))


def _sanitize_sampling_probabilities(values: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build stable sampling probabilities from non-negative scores.

    - Replace NaN/Inf with 0.
    - Clamp negatives to 0.
    - If total mass is zero/non-finite, fall back to uniform.
    - If non-zero mass count is smaller than k (without-replacement sampling),
      fall back to uniform.
    """
    probs = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(0.0)
    n = int(probs.numel())
    if n < 1:
        raise ValueError("Cannot build probabilities for an empty tensor.")

    total = probs.sum()
    if (not torch.isfinite(total)) or float(total.item()) <= 0.0:
        return torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=probs.device)

    probs = probs / total
    if int((probs > 0).sum().item()) < int(k):
        return torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=probs.device)
    return probs


@torch.no_grad()
def cur_deim_gpu(W: torch.Tensor,
                 r: int,
                 use_lowrank: bool = True,
                 oversample: int = 20,
                 niter: int = 2,
                 importance_order: str = "high") -> tuple[list[int], list[int]]:

    # --- SVD (thin or low-rank) ---
    if use_lowrank:
        # randomized / block Lanczos SVD (GPU support)
        # print(f"DEIM-CUR: SVD lowrank ({W.device})")
        qmax = min(W.shape[0], W.shape[1])
        q = min(r + oversample, qmax)

        U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
        # U, S, V = _svd_lowrank_safe(W, q=q, niter=niter, max_retries=2)

        U, V = U[:, :r], V[:, :r]
    else:
        # thin-SVD
        # print(f"DEIM-CUR: SVD ({W.device})")
        U, S, Vh = LA.svd(W, full_matrices=False)
        U, V = U[:, :r], Vh.T[:, :r]

    if importance_order not in ("high", "low"):
        raise ValueError(f"importance_order must be one of ['high', 'low'], got: {importance_order}")

    # --- DEIM Selection ---
    m, n = W.shape
    irow = torch.empty(r, dtype=torch.long, device=W.device)
    icol = torch.empty(r, dtype=torch.long, device=W.device)
    mask_r = torch.zeros(m, dtype=torch.bool, device=W.device)
    mask_c = torch.zeros(n, dtype=torch.bool, device=W.device)

    for k in range(r):
        u_abs = U[:, k].abs()
        v_abs = V[:, k].abs()

        if importance_order == "high":
            u_vec = u_abs.masked_fill(mask_r, -1.0)
            v_vec = v_abs.masked_fill(mask_c, -1.0)
            row_k = torch.argmax(u_vec)
            col_k = torch.argmax(v_vec)
        else:
            u_vec = u_abs.masked_fill(mask_r, torch.inf)
            v_vec = v_abs.masked_fill(mask_c, torch.inf)
            row_k = torch.argmin(u_vec)
            col_k = torch.argmin(v_vec)

        irow[k] = row_k
        icol[k] = col_k
        mask_r[row_k] = True
        mask_c[col_k] = True

        if k + 1 < r:
            alpha_r = U[row_k, :k+1]            # (k+1,)
            alpha_c = V[col_k, :k+1]            # (k+1,)

            denom_r = (alpha_r @ alpha_r).clamp_min(1e-12)
            denom_c = (alpha_c @ alpha_c).clamp_min(1e-12)
            U[:, k+1:] -= (U[:, :k+1] @ alpha_r.unsqueeze(1)) / denom_r
            V[:, k+1:] -= (V[:, :k+1] @ alpha_c.unsqueeze(1)) / denom_c

    return irow.tolist(), icol.tolist()


def select_rows_and_columns(
    W, A, num_rows, num_cols,
    aux_mode: str = 'wanda',
    cur_mode: str = 'deim',
    deim_importance_order: str = "high",
):
    # if num_cols != num_rows:
    #     raise ValueError("Not a square matrix.")
    m, n = W.shape
    r = min(num_rows, num_cols, m, n)

    # Fast-out
    if cur_mode == 'random':
        k_rows = min(num_rows, m)
        k_cols = min(num_cols, n)
        row_indices = torch.randperm(m, device=W.device)[:k_rows].tolist()
        col_indices = torch.randperm(n, device=W.device)[:k_cols].tolist()
        return row_indices, col_indices

    # AUX

    if aux_mode == 'wanda':
        act = A.view(1, -1).to(W.device, dtype=W.dtype)
        S = W.abs() * act  # Hadamard multiplication

    elif aux_mode == 'weight':
        S = W.abs()

    elif aux_mode == 'cov_fast':
        scale = A.view(1, -1).to(W.device, dtype=W.dtype)
        S = W * scale

    elif aux_mode == 'cov':
        Sigma = A.to(W.device, dtype=W.dtype)
        D = _matrix_sqrt_psd(Sigma)
        S = W @ D

    # CUR

    if cur_mode == 'deim':
        row_indices, col_indices = cur_deim_gpu(
            S, r,
            use_lowrank=True,
            importance_order=deim_importance_order,
        )  # GPU
        return row_indices, col_indices

    elif cur_mode == 'deim_full':
        row_indices, col_indices = cur_deim_gpu(
            S, r,
            use_lowrank=False,
            importance_order=deim_importance_order,
        )  # GPU
        return row_indices, col_indices

    elif cur_mode == 'magnitude':
        # Magnitude (Prob) with stability guards

        col_norms = torch.norm(S, p=2, dim=0)  # (n,)
        row_norms = torch.norm(S, p=2, dim=1)  # (m,)

        k_cols = min(num_cols, col_norms.numel())
        k_rows = min(num_rows, row_norms.numel())
        if k_cols < 1 or k_rows < 1:
            return [], []

        # Probabilistic sampling without replacement (normalized)
        col_prob = _sanitize_sampling_probabilities(col_norms, k_cols)
        row_prob = _sanitize_sampling_probabilities(row_norms, k_rows)

        col_indices = torch.multinomial(
            col_prob, num_samples=k_cols, replacement=False)
        row_indices = torch.multinomial(
            row_prob, num_samples=k_rows, replacement=False)
        return row_indices.tolist(), col_indices.tolist()

        # # Magnitude
        # # Sum over out_features
        # col_importance = S.sum(dim=0)
        # num_cols = min(num_cols, col_importance.size(0))
        # col_indices = torch.topk(col_importance, num_cols, largest=True)[1]
        # # Sum over in_features
        # row_importance = S.sum(dim=1)
        # num_rows = min(num_rows, row_importance.size(0))
        # row_indices = torch.topk(row_importance, num_rows, largest=True)[1]
        # # return
        # return row_indices.tolist(), col_indices.tolist()


def cur_decomposition(W, A, num_rows, num_cols, aux_mode: str = 'wanda', cur_mode: str = 'deim', use_float64: bool = True):
    orig_dtype = W.dtype
    if use_float64 and orig_dtype != torch.float64:
        W = W.to(torch.float64)
        if A is not None:
            A = A.to(torch.float64)

    row_indices, col_indices = select_rows_and_columns(
        W, A, num_rows, num_cols,
        aux_mode=aux_mode, cur_mode=cur_mode)

    C = W[:, col_indices]
    R = W[row_indices, :]

    rc = 1e-12 if orig_dtype == torch.float64 else 1e-6
    # rc = None

    if aux_mode == 'wanda':
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    elif aux_mode == 'cov_fast':
        # A: (n,) = sqrt(E[x^2]) expected
        if A.dim() != 1 or A.numel() != W.shape[1]:
            raise ValueError(
                f"[cov_fast] scale vector shape mismatch: expected ({W.shape[1]},) got {tuple(A.shape)}")
        Dvec = A.to(W.device, dtype=W.dtype)           # (n,)
        WD = W * Dvec.view(1, -1)                      # (m,n)
        RD = R * Dvec.view(1, -1)                      # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    elif aux_mode == 'cov':
        # A: (n,n) = Cov[x] expected
        if A.dim() != 2 or A.shape[0] != W.shape[1] or A.shape[1] != W.shape[1]:
            raise ValueError(
                f"[cov] covariance shape mismatch: expected ({W.shape[1]},{W.shape[1]}) got {tuple(A.shape)}")
        Sigma = A.to(W.device, dtype=W.dtype)          # (n,n)
        D = _matrix_sqrt_psd(Sigma)                    # Cov^{1/2}
        WD = W @ D                                     # (m,n)
        RD = R @ D                                     # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    else:
        # raise ValueError(f"Unknown aux_mode: {aux_mode}")
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    if use_float64 and orig_dtype != torch.float64:
        C = C.to(orig_dtype)
        R = R.to(orig_dtype)
        U = U.to(orig_dtype)

    return C, U, R, row_indices, col_indices


def cur_decomposition_fixed(
    W,
    A,
    row_indices,
    col_indices,
    aux_mode: str = 'wanda',
    use_float64: bool = True,
):
    orig_dtype = W.dtype
    if use_float64 and orig_dtype != torch.float64:
        W = W.to(torch.float64)
        if A is not None:
            A = A.to(torch.float64)

    row_indices = list(row_indices)
    col_indices = list(col_indices)

    C = W[:, col_indices]
    R = W[row_indices, :]

    rc = 1e-12 if orig_dtype == torch.float64 else 1e-6

    if aux_mode == 'wanda':
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    elif aux_mode == 'cov_fast':
        if A is None or A.dim() != 1 or A.numel() != W.shape[1]:
            raise ValueError(
                f"[cov_fast] scale vector shape mismatch: expected ({W.shape[1]},) got {None if A is None else tuple(A.shape)}")
        Dvec = A.to(W.device, dtype=W.dtype)           # (n,)
        WD = W * Dvec.view(1, -1)                      # (m,n)
        RD = R * Dvec.view(1, -1)                      # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    elif aux_mode == 'cov':
        if A is None or A.dim() != 2 or A.shape[0] != W.shape[1] or A.shape[1] != W.shape[1]:
            raise ValueError(
                f"[cov] covariance shape mismatch: expected ({W.shape[1]},{W.shape[1]}) got {None if A is None else tuple(A.shape)}")
        Sigma = A.to(W.device, dtype=W.dtype)          # (n,n)
        D = _matrix_sqrt_psd(Sigma)                    # Cov^{1/2}
        WD = W @ D                                     # (m,n)
        RD = R @ D                                     # (r,n)
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ WD
            @ torch.linalg.pinv(RD, rcond=rc)
        )

    else:
        U = (
            torch.linalg.pinv(C, rcond=rc)
            @ W
            @ torch.linalg.pinv(R, rcond=rc)
        )

    if use_float64 and orig_dtype != torch.float64:
        C = C.to(orig_dtype)
        R = R.to(orig_dtype)
        U = U.to(orig_dtype)

    return C, U, R


@torch.no_grad()
def energy_rank(W: torch.Tensor,
                A: torch.Tensor,
                aux_mode: str,
                energy: float = 0.98,
                use_lowrank: bool = True,
                niter: int = 2) -> int:
    m, n = W.shape

    # Build selection/weighted matrix
    if aux_mode == 'wanda':
        if A.dim() != 1 or A.numel() != n:
            raise ValueError(
                f"[wanda] aux vector shape mismatch: expected ({n},) got {tuple(A.shape)}")
        act = A.view(1, -1).to(W.device, dtype=W.dtype)
        M = W.abs() * act  # S = |W| * act

    elif aux_mode == 'weight':
        M = W.abs()

    elif aux_mode == 'cov_fast':
        if A.dim() != 1 or A.numel() != n:
            raise ValueError(
                f"[cov_fast] scale vector shape mismatch: expected ({n},) got {tuple(A.shape)}")
        scale = A.view(1, -1).to(W.device, dtype=W.dtype)
        M = W * scale  # M = W * sqrt(E[x^2])

    elif aux_mode == 'cov':
        if A.dim() != 2 or A.shape[0] != n or A.shape[1] != n:
            raise ValueError(
                f"[cov] covariance shape mismatch: expected ({n},{n}) got {tuple(A.shape)}")
        Sigma = A.to(W.device, dtype=W.dtype)
        D = _matrix_sqrt_psd(Sigma)  # Cov^{1/2}
        M = W @ D  # M = W @ Cov^{1/2}

    else:
        raise ValueError(f"Unknown aux_mode: {aux_mode}")

    # Energy-based rank from singular values of M
    if use_lowrank:
        q = min(
            max(256, int(min(m, n) * 0.25)),  # TODO: max 25% or 256
            min(m, n)
        )
        _, sv, _ = torch.svd_lowrank(M.float(), q=q, niter=niter)
    else:
        # SLOW
        # sv = torch.linalg.svdvals(M.float())  # descending
        _, sv, _ = torch.linalg.svd(M.float(), full_matrices=False)

    if sv.numel() == 0:
        r = 1
    else:
        e = sv.square()
        total = e.sum()
        if total <= 0 or not torch.isfinite(total):
            r = 1
        else:
            cume = torch.cumsum(e, dim=0) / total
            target = float(energy)
            # keep target within (0,1)
            if not (0.0 < target < 1.0):
                target = max(1e-6, min(target, 0.999999))
            r = int(torch.searchsorted(cume, torch.tensor(
                target, device=cume.device)).item()) + 1

    # Round up to the nearest multiple of 128
    # This can improve hardware efficiency (e.g., tensor cores).
    # We round up to preserve at least the energy target.
    if r > 0:
        r = ((r + 127) // 128) * 128

    # Guards
    r = max(1, min(r, min(m, n)))
    return r


def calculate_rank(m, n):
    """
    Calculate the rank for CUR decomposition based on matrix dimensions m and n.
    """
    try:
        r = int((math.sqrt(m**2 + 6 * m * n + n**2) - (m + n)) / 2)
    except ValueError:
        # This can happen if the term inside sqrt is negative, though unlikely with m,n > 0
        r = min(m, n)

    # Round down to the nearest multiple of 128
    # We round down to stay below the parameter breakeven point.
    if r > 0:
        r = (r // 128) * 128

    # Guards
    r = max(1, min(r, min(m, n)))
    return r


def apply_cur_to_matrix(weight, aux_info,
                        max_rank=None, min_rank=None,
                        aux_mode: str = 'wanda', cur_mode: str = 'deim',
                        energy: float | None = None,
                        row_indices=None, col_indices=None):  # 0.98
    """
    Apply CUR decomposition to a single weight matrix using WANDA metrics.

    Args:
        weight          : (m,n) matrix W
        aux_info        : (n,) for 'wanda'/'cov_fast', (n,n) for 'cov'
        min_rank        : optional rank floor
        aux_mode        : 'wanda' | 'cov_fast' | 'cov'
        energy          : retained energy ratio for rank selection (None → use size heuristic)
    """
    m, n = weight.shape
    if row_indices is not None and col_indices is not None:
        row_indices = list(row_indices)
        col_indices = list(col_indices)
        rank = min(len(row_indices), len(col_indices))
        C, U, R = cur_decomposition_fixed(
            weight,
            aux_info,
            row_indices=row_indices,
            col_indices=col_indices,
            aux_mode=aux_mode,
        )
        return C, U, R, rank, row_indices, col_indices
    if energy is not None:
        rank = energy_rank(
            weight,
            aux_info,
            aux_mode=aux_mode,
            energy=energy,
            use_lowrank=False,
        )
        upper_bound_rank = calculate_rank(m, n)
        if rank > upper_bound_rank:
            raise ValueError("No compression.")
    else:
        rank = calculate_rank(m, n)

    if max_rank:
        rank = min(rank, int(max_rank))
    if min_rank:
        rank = max(rank, int(min_rank))

    # TODO: calculate_rank here (check negative sizing)
    # TODO: skip

    C, U, R, row_indices, col_indices = cur_decomposition(
        weight, aux_info, num_rows=rank, num_cols=rank,
        aux_mode=aux_mode, cur_mode=cur_mode)
    return C, U, R, rank, row_indices, col_indices
