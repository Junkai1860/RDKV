"""
knapsack_solver.py

Lagrangian knapsack solver for mixed-precision KV cache bit-width allocation.
Given per-token importance scores and a bit budget, assigns each token a
bit-width in {0, 2, 4, 8, 16} that minimises total distortion subject to
the average-bit constraint.

Usage (smoke test):
    python knapsack_solver.py
"""

import json
import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch

# ── Default epsilon(b): normalised quantisation distortion ────────────
# Obtained from calibrate_epsilon.py on Llama-3.1-8B-Instruct / LongBench.
# epsilon(b) = (K_mean + V_mean) / 2
DEFAULT_EPSILON: Dict[int, float] = {
    0:  1.0,
    2:  0.25,
    4:  0.011,
    8:  0.000057,
    16: 0.0,
}

# Per-channel normalised MSE for K (mean over channels of per-channel MSE/var).
# Verified via per-channel calibration (job 37890908), rel_diff vs global: 2b=44.8%, 4b=44.4%, 8b=7.8%
DEFAULT_EPSILON_K: Dict[int, float] = {0: 1.0, 2: 0.277, 4: 0.011, 8: 0.000061, 16: 0.0}
DEFAULT_EPSILON_V: Dict[int, float] = {0: 1.0, 2: 0.312, 4: 0.0138, 8: 0.000048, 16: 0.0}

BITS = torch.tensor([0, 2, 4, 8, 16], dtype=torch.float32)


def knapsack_bit_allocation(
    scores: torch.Tensor,
    budget_per_token: float,
    epsilon: Optional[Dict[int, float]] = None,
    max_iters: int = 64,
    tol: float = 0.01,
    bit_options: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lagrangian relaxation knapsack solver for bit-width allocation.

    Args:
        scores:           [num_tokens] importance scores (higher = more important).
        budget_per_token:  target average bits per token (e.g. 2.0).
        epsilon:          {bit_width: distortion} map. Defaults to DEFAULT_EPSILON.
        max_iters:        max bisection iterations.
        tol:              relative tolerance on budget constraint.

    Returns:
        [num_tokens] LongTensor with values in {0, 2, 4, 8, 16}.
    """
    if epsilon is None:
        epsilon = DEFAULT_EPSILON

    bits = bit_options if bit_options is not None else BITS

    scores_ = scores.detach().float()
    device = scores_.device
    num_tokens = scores_.shape[0]

    # eps_vec: [N] distortion for each bit-width option
    eps_vec = torch.tensor(
        [epsilon[int(b)] for b in bits.tolist()], dtype=torch.float32, device=device
    )
    bits = bits.to(device=device, dtype=torch.float32)

    # cost(t, b) = score_t * eps(b) + lambda * b
    # For each token, pick b that minimises cost.
    # Binary search over lambda to satisfy budget.

    def solve(lam: float) -> torch.Tensor:
        # cost: [T, N] = scores[:,None] * eps[None,:] + lam * bits[None,:]
        cost = scores_.unsqueeze(1) * eps_vec.unsqueeze(0) + lam * bits.unsqueeze(0)
        chosen_idx = cost.argmin(dim=1)  # [T]
        return bits[chosen_idx]

    # lambda = 0 → everyone picks max-bit (max quality)
    # lambda → inf → everyone picks min-bit (min cost)
    # We need to find lambda such that avg bits ≈ budget.

    lam_lo = 0.0
    lam_hi = float(scores_.max()) * 0.375

    # Ensure lam_hi is large enough to push all tokens to min-bit
    bits_at_hi = solve(lam_hi)
    while bits_at_hi.mean().item() > budget_per_token and lam_hi < 1e12:
        lam_hi *= 2.0
        bits_at_hi = solve(lam_hi)

    # If even lam=0 gives avg_bits <= budget, return all-max-bit
    bits_at_lo = solve(lam_lo)
    if bits_at_lo.mean().item() <= budget_per_token:
        return bits_at_lo.long()

    for _ in range(max_iters):
        lam_mid = (lam_lo + lam_hi) / 2.0
        bits_mid = solve(lam_mid)
        avg_bits = bits_mid.mean().item()

        if abs(avg_bits - budget_per_token) / max(budget_per_token, 1e-8) < tol:
            return bits_mid.long()

        if avg_bits > budget_per_token:
            # Need higher lambda to reduce bits
            lam_lo = lam_mid
        else:
            # Need lower lambda to increase bits
            lam_hi = lam_mid

    # Return best result after max_iters
    return bits_mid.long()


def knapsack_bit_allocation_batched(
    scores: torch.Tensor,
    budget_per_token: Union[float, torch.Tensor],
    epsilon: Optional[Dict[int, float]] = None,
    max_iters: int = 64,
    tol: float = 0.01,
    bit_options: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched Lagrangian knapsack solver — same algorithm as
    ``knapsack_bit_allocation`` but vectorised over a batch dimension so a
    single call replaces the per-row Python loop. Saves ~300ms per layer
    of 128K prefill on Llama-3.1-8B (B=1024, H_kv=8) by collapsing 8 small
    GPU launches into 1 batched launch with the same compute shape.

    Args:
        scores:           [B, T] importance scores (one row per knapsack).
        budget_per_token: float (shared budget) or [B] tensor (per-row).
        epsilon, max_iters, tol, bit_options: same as scalar version.

    Returns:
        [B, T] LongTensor with values in {0, 2, 4, 8, 16}.

    Bit-exactness: produces results identical to applying
    ``knapsack_bit_allocation`` row-by-row. Per-row early-freeze: when a row
    converges (rel_err < tol) its bits_mid at that iteration is captured
    into final_bits and the bisection no longer updates that row's value
    even though λ_lo/λ_hi continue to be updated. Matches scalar's
    ``return bits_mid.long()`` early return semantics.
    """
    if epsilon is None:
        epsilon = DEFAULT_EPSILON

    bits = bit_options if bit_options is not None else BITS

    assert scores.dim() == 2, (
        f"knapsack_bit_allocation_batched expects [B, T]; got {tuple(scores.shape)}"
    )
    scores_ = scores.detach().float()
    device = scores_.device
    B, T = scores_.shape

    if torch.is_tensor(budget_per_token):
        assert budget_per_token.shape == (B,), (
            f"budget_per_token tensor must be [B={B}]; got {tuple(budget_per_token.shape)}"
        )
        budget = budget_per_token.detach().float().to(device)
    else:
        budget = torch.full((B,), float(budget_per_token), dtype=torch.float32, device=device)

    # eps_vec: [N] distortion for each bit-width option
    eps_vec = torch.tensor(
        [epsilon[int(b)] for b in bits.tolist()], dtype=torch.float32, device=device
    )
    bits_dev = bits.to(device=device, dtype=torch.float32)

    # cost(b, t, n) = scores[b, t] * eps[n] + lam[b] * bits[n]
    # For each (b, t), pick n that minimises cost.
    def solve_batch(lam: torch.Tensor) -> torch.Tensor:
        # scores_: [B, T], eps_vec: [N], bits_dev: [N], lam: [B]
        cost = (
            scores_.unsqueeze(2) * eps_vec.view(1, 1, -1)
            + lam.view(-1, 1, 1) * bits_dev.view(1, 1, -1)
        )  # [B, T, N]
        chosen_idx = cost.argmin(dim=2)  # [B, T]
        return bits_dev[chosen_idx]  # [B, T] float

    # Per-row bounds: lam_lo = 0, lam_hi grows until all rows hit min-bit
    lam_lo = torch.zeros(B, device=device)
    lam_hi = scores_.amax(dim=1) * 0.375
    lam_hi = lam_hi.clamp_min(1e-8)  # avoid lam_hi=0 when all scores==0

    # Expand lam_hi until each row's avg_bits at lam_hi <= budget.
    # Vectorised version of the scalar-loop "while bits_at_hi.mean() > budget: lam_hi *= 2".
    for _ in range(40):  # cap doubling steps; matches scalar 1e12 cutoff
        bits_at_hi = solve_batch(lam_hi)
        avg_at_hi = bits_at_hi.mean(dim=1)
        need_higher = avg_at_hi > budget
        if not need_higher.any():
            break
        lam_hi = torch.where(need_higher, lam_hi * 2.0, lam_hi)
        if (lam_hi >= 1e12).all():
            break

    # Row-wise early return: if avg at lam=0 already <= budget, use those bits.
    bits_at_lo = solve_batch(lam_lo)
    avg_at_lo = bits_at_lo.mean(dim=1)  # [B]
    converged_at_lo = avg_at_lo <= budget  # [B]

    # Bisection with per-row early-freeze: when a row converges (rel_err < tol)
    # we capture its bits_mid at that iteration and stop updating it. This
    # exactly mirrors the scalar version's `return bits_mid.long()` early
    # return — preserving bit-exactness on a per-row basis even when other
    # rows in the batch keep iterating.
    final_bits = bits_at_lo.clone()
    converged = converged_at_lo.clone()
    bits_mid = bits_at_lo.clone()  # init for rows that never converge
    for _ in range(max_iters):
        lam_mid = (lam_lo + lam_hi) / 2.0
        bits_mid = solve_batch(lam_mid)
        avg_mid = bits_mid.mean(dim=1)  # [B]

        # Per-row convergence check; relative error vs. budget
        rel_err = (avg_mid - budget).abs() / budget.clamp(min=1e-8)
        just_converged = (rel_err < tol) & (~converged)
        if just_converged.any():
            final_bits = torch.where(
                just_converged.view(-1, 1), bits_mid, final_bits
            )
            converged = converged | just_converged

        if converged.all().item():
            break

        # Per-row bound update (rows already converged stay frozen because
        # final_bits no longer reads from bits_mid for them).
        too_high = avg_mid > budget  # [B] bool — need higher lam
        lam_lo = torch.where(too_high, lam_mid, lam_lo)
        lam_hi = torch.where(too_high, lam_hi, lam_mid)

    # Rows that never converged in max_iters: fall back to last bits_mid
    # (matches scalar path's "return best result after max_iters" branch).
    final_bits = torch.where(
        converged.view(-1, 1),
        final_bits,
        bits_mid,
    )

    return final_bits.long()


def allocate_v_bits_with_fp16_topk(
    scores: torch.Tensor,
    budget_per_token: float,
    fp16_topk: int = 0,
    epsilon: Optional[Dict[int, float]] = None,
    max_iters: int = 64,
    tol: float = 0.01,
    bit_options: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reserve top-k tokens for FP16, then run knapsack on the remainder.

    The solver budget is interpreted as average bits per token. This helper
    converts that into a total V budget, deducts the forced-FP16 reservation,
    and then computes a new average budget for the remaining tokens.

    Returns:
        (v_bits, diagnostics)
        v_bits: [num_tokens] LongTensor with values in {0, 2, 4, 8, 16}.
        diagnostics: metadata for logging / validation.
    """
    scores_ = scores.detach().float()
    device = scores_.device
    num_tokens = int(scores_.shape[0])
    requested_topk = max(int(fp16_topk), 0)

    diagnostics: Dict[str, Any] = {
        "requested_fp16_topk": requested_topk,
        "applied_fp16_topk": 0,
        "preallocation_applied": False,
        "used_budget_per_token": float(budget_per_token),
        "adjusted_budget_per_token": float(budget_per_token),
        "total_v_bits": float(budget_per_token) * num_tokens,
        "reserved_fp16_bits": 0.0,
        "remaining_v_bits": float(budget_per_token) * num_tokens,
        "fallback_reason": None,
    }

    if requested_topk <= 0 or num_tokens == 0:
        bits = knapsack_bit_allocation(
            scores_,
            budget_per_token=budget_per_token,
            epsilon=epsilon,
            max_iters=max_iters,
            tol=tol,
            bit_options=bit_options,
        )
        diagnostics["actual_fp16_count"] = int((bits == 16).sum().item())
        return bits, diagnostics

    applied_topk = min(requested_topk, num_tokens)
    total_v_bits = float(budget_per_token) * num_tokens
    reserved_fp16_bits = applied_topk * 16.0
    remaining_v_bits = total_v_bits - reserved_fp16_bits
    remaining_count = num_tokens - applied_topk

    diagnostics["applied_fp16_topk"] = applied_topk
    diagnostics["reserved_fp16_bits"] = reserved_fp16_bits
    diagnostics["remaining_v_bits"] = remaining_v_bits

    if remaining_count <= 0:
        warnings.warn(
            f"fp16_topk={requested_topk} covers all {num_tokens} tokens; "
            "falling back to the original V-side knapsack allocation."
        )
        bits = knapsack_bit_allocation(
            scores_,
            budget_per_token=budget_per_token,
            epsilon=epsilon,
            max_iters=max_iters,
            tol=tol,
            bit_options=bit_options,
        )
        diagnostics["fallback_reason"] = "topk_covers_all_tokens"
        diagnostics["actual_fp16_count"] = int((bits == 16).sum().item())
        return bits, diagnostics

    if remaining_v_bits <= 0:
        warnings.warn(
            f"fp16_topk={requested_topk} exhausts the V budget "
            f"({reserved_fp16_bits:.1f} > {total_v_bits:.1f}); "
            "falling back to the original V-side knapsack allocation."
        )
        bits = knapsack_bit_allocation(
            scores_,
            budget_per_token=budget_per_token,
            epsilon=epsilon,
            max_iters=max_iters,
            tol=tol,
            bit_options=bit_options,
        )
        diagnostics["fallback_reason"] = "fp16_reservation_exhausts_budget"
        diagnostics["actual_fp16_count"] = int((bits == 16).sum().item())
        return bits, diagnostics

    adjusted_budget_per_token = remaining_v_bits / remaining_count
    topk_indices = torch.topk(scores_, applied_topk).indices
    topk_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    topk_mask[topk_indices] = True
    remaining_scores = scores_[~topk_mask]

    remaining_bits = knapsack_bit_allocation(
        remaining_scores,
        budget_per_token=adjusted_budget_per_token,
        epsilon=epsilon,
        max_iters=max_iters,
        tol=tol,
        bit_options=bit_options,
    )

    v_bits = torch.empty(num_tokens, dtype=remaining_bits.dtype, device=device)
    v_bits[topk_mask] = 16
    v_bits[~topk_mask] = remaining_bits

    diagnostics["preallocation_applied"] = True
    diagnostics["adjusted_budget_per_token"] = adjusted_budget_per_token
    diagnostics["actual_fp16_count"] = int((v_bits == 16).sum().item())
    return v_bits.long(), diagnostics


def load_epsilon_from_calibration(path: str) -> Dict[int, float]:
    """Load epsilon(b) from a calibration JSON file.

    For each bit-width, takes (K_mean + V_mean) / 2 as the distortion estimate.
    """
    with open(path, "r") as f:
        data = json.load(f)
    eps = {}
    for b_str, stats in data.items():
        b = int(b_str)
        eps[b] = (stats["K_mean"] + stats["V_mean"]) / 2.0
    return eps


def load_epsilon_kv_from_calibration(path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load separate K and V epsilon(b) from a calibration JSON file.

    Returns:
        (epsilon_K, epsilon_V) where each is {bit_width: distortion}.
        epsilon_K prefers K_per_channel_mean (per-channel NMSE) when available,
        otherwise falls back to K_mean (global NMSE).
        epsilon_V uses V_mean for {0,2,4,8,16}.
    """
    with open(path, "r") as f:
        data = json.load(f)
    eps_k: Dict[int, float] = {0: 1.0}  # channel eviction
    eps_v: Dict[int, float] = {}
    for b_str, stats in data.items():
        b = int(b_str)
        if b in {2, 4, 8}:
            eps_k[b] = stats.get("K_per_channel_mean", stats["K_mean"])
        if b in {0, 2, 4, 8, 16}:
            eps_v[b] = stats["V_mean"]
    eps_k[16] = 0.0
    return eps_k, eps_v


def knapsack_solve_kv_split(
    token_scores: torch.Tensor,
    channel_scores: torch.Tensor,
    budget_per_token: float,
    epsilon_K: Optional[Dict[int, float]] = None,
    epsilon_V: Optional[Dict[int, float]] = None,
    k_budget_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve separate knapsack problems for K (per-channel) and V (per-token).

    Args:
        token_scores:    [seq_len] per-token importance scores for V.
        channel_scores:  [head_dim] per-channel importance scores for K.
        budget_per_token: K+V combined average bits per element.
        epsilon_K:       {bit: distortion} for K. Defaults to DEFAULT_EPSILON_K.
        epsilon_V:       {bit: distortion} for V. Defaults to DEFAULT_EPSILON_V.
        k_budget_ratio:  fraction of total budget allocated to K (default 0.5).

    Returns:
        (k_bits [head_dim], v_bits [seq_len]) with K bits in {0,2,4,8,16}
        and V bits in {0,2,4,8,16}.  K bits may include 0 (channel eviction).
    """
    if epsilon_K is None:
        epsilon_K = DEFAULT_EPSILON_K
    if epsilon_V is None:
        epsilon_V = DEFAULT_EPSILON_V

    k_avg = k_budget_ratio * budget_per_token * 2
    v_avg = (1 - k_budget_ratio) * budget_per_token * 2

    k_bit_options = torch.tensor([2, 4, 8, 16], dtype=torch.float32)
    k_bits = knapsack_bit_allocation(
        channel_scores, k_avg, epsilon=epsilon_K, bit_options=k_bit_options,
    )
    v_bits = knapsack_bit_allocation(
        token_scores, v_avg, epsilon=epsilon_V,
    )
    return k_bits, v_bits


# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Knapsack Solver Smoke Test ===\n")

    torch.manual_seed(42)

    # Synthetic log-normal scores (1000 tokens)
    scores = torch.exp(torch.randn(1000) * 1.5 + 2.0)  # log-normal
    print(f"Score stats: min={scores.min():.2f} max={scores.max():.2f} "
          f"mean={scores.mean():.2f} std={scores.std():.2f}\n")

    budgets = [1.0, 2.0, 3.0, 4.0, 6.0]
    all_ok = True

    for budget in budgets:
        bits = knapsack_bit_allocation(scores, budget_per_token=budget)
        avg_bits = bits.float().mean().item()
        dist = {int(b): int((bits == b).sum()) for b in [0, 2, 4, 8, 16]}
        rel_err = abs(avg_bits - budget) / budget

        status = "OK" if rel_err < 0.05 else "FAIL"
        if rel_err >= 0.05:
            all_ok = False
        print(f"  budget={budget:.1f}  actual_avg={avg_bits:.3f}  "
              f"rel_err={rel_err:.4f}  [{status}]  dist={dist}")

    # Verify high-score tokens get higher bit-widths
    print("\n--- Monotonicity check ---")
    bits_2 = knapsack_bit_allocation(scores, budget_per_token=2.0)
    top50_idx = scores.argsort(descending=True)[:50]
    bot50_idx = scores.argsort()[:50]
    top50_avg = bits_2[top50_idx].float().mean().item()
    bot50_avg = bits_2[bot50_idx].float().mean().item()
    mono_ok = top50_avg >= bot50_avg
    if not mono_ok:
        all_ok = False
    print(f"  Top-50 score tokens avg bit: {top50_avg:.1f}")
    print(f"  Bottom-50 score tokens avg bit: {bot50_avg:.1f}")
    print(f"  Monotonicity: {'OK' if mono_ok else 'FAIL'}")

    print("\n--- FP16 top-k pre-allocation ---")
    base_bits = knapsack_bit_allocation(scores, budget_per_token=2.0)
    topk0_bits, topk0_diag = allocate_v_bits_with_fp16_topk(scores, budget_per_token=2.0, fp16_topk=0)
    nochange_ok = torch.equal(base_bits, topk0_bits)
    if not nochange_ok:
        all_ok = False
    print(f"  fp16_topk=0 parity: {'OK' if nochange_ok else 'FAIL'} "
          f"(actual_fp16={topk0_diag['actual_fp16_count']})")

    forced_bits, forced_diag = allocate_v_bits_with_fp16_topk(scores, budget_per_token=6.0, fp16_topk=10)
    top10_idx = scores.argsort(descending=True)[:10]
    forced_ok = bool((forced_bits[top10_idx] == 16).all().item())
    if not forced_ok:
        all_ok = False
    print(f"  fp16_topk=10 forced top-10: {'OK' if forced_ok else 'FAIL'} "
          f"(actual_fp16={forced_diag['actual_fp16_count']}, "
          f"adjusted_budget={forced_diag['adjusted_budget_per_token']:.3f})")

    fallback_bits, fallback_diag = allocate_v_bits_with_fp16_topk(scores, budget_per_token=0.1, fp16_topk=1000)
    fallback_ok = fallback_diag["fallback_reason"] is not None
    if not fallback_ok:
        all_ok = False
    print(f"  oversized reservation fallback: {'OK' if fallback_ok else 'FAIL'} "
          f"({fallback_diag['fallback_reason']})")

    # K/V split solver test
    print("\n--- K/V Split Solver ---")
    token_scores = torch.exp(torch.randn(1000) * 1.5 + 2.0)
    channel_scores = torch.exp(torch.randn(128) * 1.0 + 1.0)
    for budget in [2.0, 4.0]:
        k_bits, v_bits = knapsack_solve_kv_split(
            token_scores, channel_scores, budget_per_token=budget,
        )
        # K bits must be in {0,2,4,8,16}
        k_valid = set(k_bits.unique().tolist()).issubset({0, 2, 4, 8, 16})
        # V bits must be in {0,2,4,8,16}
        v_valid = set(v_bits.unique().tolist()).issubset({0, 2, 4, 8, 16})
        k_avg = k_bits.float().mean().item()
        v_avg = v_bits.float().mean().item()
        combined_avg = (k_avg + v_avg) / 2
        budget_ok = abs(combined_avg - budget) / budget < 0.15
        split_ok = k_valid and v_valid and budget_ok
        if not split_ok:
            all_ok = False
        k_dist = {int(b): int((k_bits == b).sum()) for b in [0, 2, 4, 8, 16]}
        v_dist = {int(b): int((v_bits == b).sum()) for b in [0, 2, 4, 8, 16]}
        print(f"  budget={budget:.1f}  k_avg={k_avg:.2f}  v_avg={v_avg:.2f}  "
              f"combined={combined_avg:.2f}  [{'OK' if split_ok else 'FAIL'}]")
        print(f"    K dist={k_dist}  V dist={v_dist}")

    # Verify fake_quantize 8-bit NMSE
    print("\n--- fake_quantize 8-bit NMSE check ---")
    try:
        from calibrate_epsilon import fake_quantize
        x = torch.randn(1, 8, 512, 128, dtype=torch.float16)
        x_hat = fake_quantize(x.float(), 8, dim=2)
        nmse = ((x.float() - x_hat) ** 2).sum() / (x.float() ** 2).sum()
        nmse_ok = nmse.item() < 1e-3
        if not nmse_ok:
            all_ok = False
        print(f"  8-bit fake_quantize NMSE: {nmse.item():.2e}  "
              f"[{'OK' if nmse_ok else 'FAIL'}]")
    except ImportError:
        print("  (skipped — calibrate_epsilon not importable without model deps)")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
