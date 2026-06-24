"""clark_lib.py - compute functions for the clark-tuning presentation notebook.

Copies of the cross-validated GP tuning, generative-model, regression, and COM
helpers from clark-tuning.ipynb, plus a few presentation-specific helpers. This
module is the single source of truth for the presentation; clark-tuning.ipynb is
left untouched.
"""
import numpy as np
from typing import Callable, Sequence, Dict, Any
from itertools import combinations
from tqdm.auto import tqdm



def gaussian_gram(
    domain: np.ndarray,
    length_scale: float,
    amplitude: float = 1.0,
):
    """
    Default squared-exponential / Gaussian kernel.
    """
    domain = np.asarray(domain, dtype=float)
    dx = domain[:, None] - domain[None, :]
    return amplitude**2 * np.exp(-0.5 * (dx / length_scale) ** 2)


def periodic_gaussian_gram(
    domain: np.ndarray,
    length_scale: float,
    amplitude: float = 1.0,
    period: float | None = None,
):
    """
    Squared-exponential / Gaussian Gram matrix on a periodic 1D domain.

    Uses circular distance:

        d_periodic(x, y) = min(|x - y|, period - |x - y|)

    and

        K(x, y) = amplitude^2 exp(-0.5 d_periodic(x, y)^2 / length_scale^2)

    Parameters
    ----------
    domain : array, shape (D,)
        Bin centres or grid points.

    length_scale : float
        Kernel length-scale in the same units as `domain`.

    amplitude : float, default=1.0
        Kernel amplitude.

    period : float, optional
        Period of the domain. If None, it is inferred assuming `domain`
        is an evenly-spaced grid of bin centres.

    Returns
    -------
    K : array, shape (D, D)
        Periodic Gaussian Gram matrix.
    """
    domain = np.asarray(domain, dtype=float)

    if domain.ndim != 1:
        raise ValueError("domain must be one-dimensional.")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")

    D = domain.size

    if period is None:
        if D <= 1:
            period = 1.0
        else:
            dx = np.median(np.diff(np.sort(domain)))
            period = domain.max() - domain.min() + dx

    delta = np.abs(domain[:, None] - domain[None, :])
    delta = np.mod(delta, period)
    circular_delta = np.minimum(delta, period - delta)

    K = amplitude**2 * np.exp(
        -0.5 * (circular_delta / length_scale) ** 2
    )

    return K


def _partition_indices(T: int, K: int, L: int):
    """
    Split time into K*L segments, then build partition k from segments

        k, k+K, k+2K, ...

    for k = 0, ..., K-1.
    """
    if K < 2:
        raise ValueError("K must be at least 2 for cross-validation.")
    if L < 1:
        raise ValueError("L must be at least 1.")
    if K * L > T:
        raise ValueError("K * L must be <= T.")

    segments = np.array_split(np.arange(T, dtype=np.int64), K * L)
    return [np.concatenate(segments[k::K]) for k in range(K)]


def _bin_behaviour_row_and_domain(b: np.ndarray, D: int):
    """
    Equal-width binning of one behavioural variable.

    Returns
    -------
    bins : (T,)
        Integer bin indices in {0, ..., D-1}; invalid values get -1.

    domain : (D,)
        Bin-centre locations in the original units of b.
    """
    b = np.asarray(b, dtype=float)
    bins = np.full(b.shape, -1, dtype=np.int64)

    valid = np.isfinite(b)
    if not np.any(valid):
        return bins, np.arange(D, dtype=float)

    lo = np.nanmin(b)
    hi = np.nanmax(b)

    if hi <= lo:
        bins[valid] = 0
        return bins, np.full(D, lo, dtype=float)

    edges = np.linspace(lo, hi, D + 1)
    domain = 0.5 * (edges[:-1] + edges[1:])

    z = (b[valid] - lo) / (hi - lo)
    bins[valid] = np.clip(np.floor(D * z).astype(np.int64), 0, D - 1)

    return bins, domain


def _binned_partition_stats(
    X: np.ndarray,
    S: np.ndarray,
    idx: np.ndarray,
    bins: np.ndarray,
    D: int,
):
    """
    Compute binned firing-rate sums, firing-rate counts, spike counts,
    and dwell counts for one behavioural variable and one partition.
    """
    N = X.shape[0]

    x_sums = np.zeros((N, D), dtype=float)
    x_counts = np.zeros((N, D), dtype=float)
    spike_counts = np.zeros((N, D), dtype=float)
    dwell_counts = np.bincount(bins, minlength=D).astype(float)

    for n in range(N):
        x = np.asarray(X[n, idx], dtype=float)
        ok = np.isfinite(x)

        if np.any(ok):
            x_sums[n] = np.bincount(
                bins[ok],
                weights=x[ok],
                minlength=D,
            )
            x_counts[n] = np.bincount(
                bins[ok],
                minlength=D,
            )

        s = np.asarray(S[n, idx], dtype=float)
        s = np.nan_to_num(s, nan=0.0)

        spike_counts[n] = np.bincount(
            bins,
            weights=s,
            minlength=D,
        )

    return x_sums, x_counts, spike_counts, dwell_counts


def _gp_map_batch(
    Y: np.ndarray,       # (M, D)
    Kmat: np.ndarray,    # (D, D)
    sigma: float,
    obs_jitter: float = 1e-10,
):
    """
    Compute GP posterior MAP curves for rows of Y.

    If Y has no NaNs:

        f = K (K + sigma^2 I)^(-1) y

    If a row has NaNs, those bins are treated as unobserved and the
    posterior mean is evaluated at all D bins.
    """
    Y = np.asarray(Y, dtype=float)
    Kmat = np.asarray(Kmat, dtype=float)

    M, D = Y.shape
    out = np.zeros((M, D), dtype=float)

    finite = np.isfinite(Y)
    I = np.eye(D, dtype=float)

    if np.all(finite):
        R = Kmat + (sigma**2 + obs_jitter) * I
        Z = np.linalg.solve(R, Y.T)       # (D, M)
        return (Kmat @ Z).T               # (M, D)

    for m in range(M):
        obs = finite[m]

        if not np.any(obs):
            out[m] = 0.0
            continue

        y_obs = Y[m, obs]

        K_oo = Kmat[np.ix_(obs, obs)]
        K_all_o = Kmat[:, obs]

        R = K_oo + (sigma**2 + obs_jitter) * np.eye(obs.sum())
        z = np.linalg.solve(R, y_obs)

        out[m] = K_all_o @ z

    return out


def _make_gram_with_lengthscale_and_amplitude(
    p: int,
    ell: float,
    amplitude: float,
    domains: Sequence[np.ndarray],
    gram_fns: Sequence[Callable],
    gram_fn_args: Sequence[Dict[str, Any]],
    length_scale_key: str = "length_scale",
    amplitude_key: str = "amplitude",
):
    """
    Construct one GP Gram matrix for behavioural variable p.

    Assumes

        gram_fns[p](domains[p], **args)

    where args includes both the length-scale and amplitude.
    """
    args = dict(gram_fn_args[p])
    args[length_scale_key] = float(ell)
    args[amplitude_key] = float(amplitude)

    return np.asarray(
        gram_fns[p](domains[p], **args),
        dtype=float,
    )


def histogram_equalised_timestep_subsample(
    idx0: np.ndarray,
    bins_p: np.ndarray,
    D: int,
    rng: np.random.Generator,
    target_per_bin: int | None = None,
    replace: bool = False,
    shuffle: bool = True,
):
    """
    Subsample timesteps within one partition so that occupied behavioural
    bins have approximately equal counts.

    Parameters
    ----------
    idx0 : array, shape (T_partition,)
        Original timestep indices for this partition.

    bins_p : array, shape (T,)
        Bin index for behavioural variable p at each timestep.
        Invalid timesteps should have bin -1.

    D : int
        Number of behavioural bins.

    rng : np.random.Generator
        Random number generator.

    target_per_bin : int, optional
        Number of timesteps to keep per occupied bin. If None, uses the
        minimum count across occupied bins, giving exact equalisation across
        non-empty bins.

    replace : bool
        If True, sample with replacement when a bin has fewer than
        target_per_bin timesteps. If False, use at most the available count.

    shuffle : bool
        If True, shuffle the final selected timesteps.

    Returns
    -------
    idx : array
        Subsampled timestep indices.

    bins : array
        Bin labels corresponding to idx.

    original_counts : array, shape (D,)
        Histogram before subsampling.

    sampled_counts : array, shape (D,)
        Histogram after subsampling.
    """
    b = bins_p[idx0]
    valid = b >= 0

    idx_valid = idx0[valid]
    bins_valid = b[valid]

    if idx_valid.size == 0:
        return (
            idx_valid,
            bins_valid,
            np.zeros(D, dtype=int),
            np.zeros(D, dtype=int),
        )

    original_counts = np.bincount(bins_valid, minlength=D)
    occupied = original_counts > 0

    if not np.any(occupied):
        return (
            np.array([], dtype=idx0.dtype),
            np.array([], dtype=bins_p.dtype),
            original_counts,
            np.zeros(D, dtype=int),
        )

    if target_per_bin is None:
        # Exact equalisation across non-empty bins.
        target_per_bin = int(original_counts[occupied].min())

    selected_positions = []

    for d in range(D):
        pos_d = np.flatnonzero(bins_valid == d)

        if pos_d.size == 0:
            continue

        if replace:
            n_take = target_per_bin
        else:
            n_take = min(target_per_bin, pos_d.size)

        if n_take <= 0:
            continue

        chosen = rng.choice(pos_d, size=n_take, replace=replace)
        selected_positions.append(chosen)

    if len(selected_positions) == 0:
        return (
            np.array([], dtype=idx0.dtype),
            np.array([], dtype=bins_p.dtype),
            original_counts,
            np.zeros(D, dtype=int),
        )

    selected_positions = np.concatenate(selected_positions)

    if shuffle:
        rng.shuffle(selected_positions)

    idx = idx_valid[selected_positions]
    bins = bins_valid[selected_positions]

    sampled_counts = np.bincount(bins, minlength=D)

    return idx, bins, original_counts, sampled_counts


def _crossvalidated_null_poisson_ll(
    lambda_null: np.ndarray,      # (N,)
    spike_counts_p: np.ndarray,   # (N, D, K)
    dwell_counts_p: np.ndarray,   # (D, K)
    k: int,
    dt: float,
):
    """
    Compute the held-out Poisson log-likelihood under a constant-rate
    null model trained on partition k.

    The null intensity for neuron n is

        lambda_null[n] 

    Then this constant intensity is evaluated on all q != k partitions,
    and averaged over held-out partitions.

    Returns
    -------
    ll_null : array, shape (N,)
        Average held-out null log-likelihood for training partition k.
    """
    N, D, K = spike_counts_p.shape

    log_lambda_null = np.log(lambda_null)

    ll_null = np.zeros(N, dtype=float)

    for q in range(K):
        if q == k:
            continue

        heldout_spikes = np.sum(spike_counts_p[:, :, q], axis=1)  # (N,)
        heldout_time = dt * np.sum(dwell_counts_p[:, q])          # scalar

        ll_null += (
            heldout_spikes * log_lambda_null
            -
            heldout_time * lambda_null
        )

    ll_null /= K - 1

    return ll_null


def gp_cross_validated_tuning_curves(
    X: np.ndarray,                          # (N, T)
    B: np.ndarray,                          # (P, T)
    S: np.ndarray,                          # (N, T)
    D: int = 100,
    K: int = 2,
    L: int = 1,
    length_scales: np.ndarray | None = None,
    amplitudes: np.ndarray | None = None,
    sigma: float = 0.1,
    dt: float = 0.025,
    domains: Sequence[np.ndarray] | None = None,
    gram_fns: Sequence[Callable] | Callable | None = None,
    gram_fn_args: Sequence[Dict[str, Any]] | Dict[str, Any] | None = None,
    length_scale_key: str = "length_scale",
    amplitude_key: str = "amplitude",
    eps_rate: float = 1e-12,
    obs_jitter: float = 0.0,
    equalise_histogram: bool = False,
    target_per_bin: int | None = None,
    equalise_replace: bool = False,
    equalise_seed: int | None = None,
    equalise_final_histogram: bool | None = None,
):
    """
    Cross-validated GP-smoothed tuning-curve estimation.

    For each behavioural variable p, time is split into K interleaved
    partitions. Within each training partition the binned firing-rate curve is
    GP-smoothed (squared-exponential MAP) for every (length_scale, amplitude)
    on the supplied grids, and the pair maximising the average held-out Poisson
    log-likelihood is selected per neuron. The final whole-dataset curve for
    each neuron uses its partition-averaged best length-scale and amplitude.
    `sigma` is a single scalar observation-noise std shared by all GP fits.

    Returns
    -------
    result : dict with keys

        F_raw : (N, P, D, K)
            Observed/raw partition tuning curves (firing-rate mean per bin).

        F_smooth : (N, P, D, K)
            GP MAP tuning curves in each partition, using that partition's
            best cross-validated length-scale and amplitude.

        F_best : (N, P, D)
            Final GP MAP tuning curves from the whole dataset, using each
            neuron's partition-averaged best length-scale and amplitude.

        F_raw_normed, F_smooth_normed, F_best_normed
            The corresponding arrays divided by their per-curve mean over the D
            bins (so each curve integrates to ~1).

        scores : (N, P)
            Per-(neuron, variable) mean over partitions of (best_ll - null_ll).
            Positive means the tuning-curve model beats the constant-rate null;
            negative means it is worse.

        best_ll : (N, P, K)
            Best held-out Poisson log-likelihood per neuron/variable/partition.

        null_ll : (N, P, K)
            Held-out log-likelihood of the constant-rate null per
            neuron/variable/partition.

        optimal_length_scales : (N, P, K)
            Best length-scale for each neuron, variable, and partition.

        optimal_amplitudes : (N, P, K)
            Best amplitude for each neuron, variable, and partition.

        avg_length_scales : (N, P)
            Average best length-scale across partitions, per neuron.

        avg_amplitudes : (N, P)
            Average best amplitude across partitions, per neuron.

        sigma : float
            The scalar observation-noise std used for all GP fits.

        domains : length-P list
            Domains (bin centres) used for the GP kernels.

        bin_idx : (P, T)
            Behavioural bin index at each timestep (-1 where invalid).
    """
    X = np.asarray(X)
    B = np.asarray(B)
    S = np.asarray(S)

    if X.ndim != 2 or B.ndim != 2 or S.ndim != 2:
        raise ValueError("X, B, and S must all be 2D arrays.")

    N, T = X.shape
    P, T_B = B.shape
    N_S, T_S = S.shape

    if T_B != T or N_S != N or T_S != T:
        raise ValueError(
            f"Expected X.shape=(N,T), B.shape=(P,T), S.shape=(N,T); "
            f"got X={X.shape}, B={B.shape}, S={S.shape}."
        )

    if length_scales is None:
        length_scales = np.linspace(1e-2, 1.0, 10)

    if amplitudes is None:
        amplitudes = np.logspace(-1, 1, 10)

    length_scales = np.asarray(length_scales, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)

    if np.any(length_scales <= 0):
        raise ValueError("All length_scales must be positive.")
    if np.any(amplitudes <= 0):
        raise ValueError("All amplitudes must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if gram_fns is None:
        gram_fns = [gaussian_gram for _ in range(P)]
    elif callable(gram_fns):
        gram_fns = [gram_fns for _ in range(P)]
    else:
        gram_fns = list(gram_fns)

    if gram_fn_args is None:
        gram_fn_args = [{} for _ in range(P)]
    elif isinstance(gram_fn_args, dict):
        gram_fn_args = [dict(gram_fn_args) for _ in range(P)]
    else:
        gram_fn_args = list(gram_fn_args)

    if len(gram_fns) != P:
        raise ValueError("gram_fns must have length P.")
    if len(gram_fn_args) != P:
        raise ValueError("gram_fn_args must have length P.")

    equalise_rng = np.random.default_rng(equalise_seed)

    if equalise_final_histogram is None:
        equalise_final_histogram = equalise_histogram

    partitions = _partition_indices(T, K, L)

    F_raw = np.full((N, P, D, K), np.nan, dtype=float)
    F_smooth = np.full((N, P, D, K), np.nan, dtype=float)
    F_best = np.full((N, P, D), np.nan, dtype=float)

    optimal_length_scales = np.full((N, P, K), np.nan, dtype=float)
    optimal_amplitudes = np.full((N, P, K), np.nan, dtype=float)

    null_ll = np.full((N, P, K), np.nan, dtype=float)
    best_ll = np.full((N, P, K), -np.inf, dtype=float)
    best_ell_idx = np.zeros((N, P, K), dtype=np.int64)
    best_amp_idx = np.zeros((N, P, K), dtype=np.int64)

    bin_idx = np.full((P, T), -1, dtype=np.int64)
    inferred_domains = []

    # If user supplies domains, use them for the GP. Otherwise infer from B.
    if domains is not None:
        domains = [np.asarray(d, dtype=float) for d in domains]
        if len(domains) != P:
            raise ValueError("domains must have length P.")
        for p in range(P):
            if domains[p].shape != (D,):
                raise ValueError(f"domains[{p}] must have shape ({D},).")

    for p in range(P):
        bins_p, domain_p = _bin_behaviour_row_and_domain(B[p], D)
        bin_idx[p] = bins_p

        if domains is None:
            inferred_domains.append(domain_p)

    if domains is None:
        domains = inferred_domains

    total_iters = P * K * len(length_scales) * len(amplitudes)

    with tqdm(total=total_iters, desc="Cross Validation", unit="param set") as pbar:
        for p in range(P):
            bins_p = bin_idx[p]

            # ------------------------------------------------------------
            # Precompute binned stats for this behavioural variable.
            # ------------------------------------------------------------
            x_sums_p = np.zeros((N, D, K), dtype=float)
            x_counts_p = np.zeros((N, D, K), dtype=float)
            spike_counts_p = np.zeros((N, D, K), dtype=float)
            dwell_counts_p = np.zeros((D, K), dtype=float)

            for k, idx0 in enumerate(partitions):
                if equalise_histogram:
                    idx, bins, original_counts, sampled_counts = (
                        histogram_equalised_timestep_subsample(
                            idx0=idx0,
                            bins_p=bins_p,
                            D=D,
                            rng=equalise_rng,
                            target_per_bin=target_per_bin,
                            replace=equalise_replace,
                            shuffle=True,
                        )
                    )
                else:
                    b = bins_p[idx0]
                    valid = b >= 0

                    idx = idx0[valid]
                    bins = b[valid]

                x_sums, x_counts, spike_counts, dwell_counts = _binned_partition_stats(
                    X=X,
                    S=S,
                    idx=idx,
                    bins=bins,
                    D=D,
                )

                x_sums_p[:, :, k] = x_sums
                x_counts_p[:, :, k] = x_counts
                spike_counts_p[:, :, k] = spike_counts
                dwell_counts_p[:, k] = dwell_counts

            Fp_raw = np.divide(
                x_sums_p,
                x_counts_p,
                out=np.full_like(x_sums_p, np.nan, dtype=float),
                where=x_counts_p > 0,
            )

            F_raw[:, p, :, :] = Fp_raw

            # ------------------------------------------------------------
            # Cache Gram matrices for this p, length-scale and amplitude.
            # K_mats[i_ell][i_amp] has shape (D, D).
            # ------------------------------------------------------------
            K_mats = [
                [
                    _make_gram_with_lengthscale_and_amplitude(
                        p=p,
                        ell=ell,
                        amplitude=amp,
                        domains=domains,
                        gram_fns=gram_fns,
                        gram_fn_args=gram_fn_args,
                        length_scale_key=length_scale_key,
                        amplitude_key=amplitude_key,
                    )
                    for amp in amplitudes
                ]
                for ell in length_scales
            ]

            # ------------------------------------------------------------
            # Cross-validation:
            # for each training partition k, choose ell and amplitude by
            # maximising average held-out Poisson log-likelihood.
            # ------------------------------------------------------------
            for k in range(K):
                Y_train = Fp_raw[:, :, k]  # (N, D)

                null_ll[:, p, k] = _crossvalidated_null_poisson_ll(
                    lambda_null=Y_train.mean(axis=1),
                    spike_counts_p=spike_counts_p,
                    dwell_counts_p=dwell_counts_p,
                    k=k,
                    dt=dt,
                )

                for ell_idx, ell in enumerate(length_scales):
                    for amp_idx, amp in enumerate(amplitudes):
                        Kmat = K_mats[ell_idx][amp_idx]

                        F_map = _gp_map_batch(
                            Y_train,
                            Kmat=Kmat,
                            sigma=float(sigma),
                            obs_jitter=obs_jitter,
                        )

                        # GP MAP is unconstrained; Poisson intensity must be positive.
                        # Only floor for likelihood evaluation.
                        rates = np.clip(F_map, eps_rate, None)
                        log_rates = np.log(rates)

                        ll = np.zeros(N, dtype=float)

                        for q in range(K):
                            if q == k:
                                continue

                            ll += (
                                np.einsum(
                                    "nd,nd->n",
                                    spike_counts_p[:, :, q],
                                    log_rates,
                                    optimize=True,
                                )
                                - dt
                                * np.einsum(
                                    "d,nd->n",
                                    dwell_counts_p[:, q],
                                    rates,
                                    optimize=True,
                                )
                            )

                        ll /= K - 1

                        update = ll > best_ll[:, p, k]

                        best_ll[update, p, k] = ll[update]
                        best_ell_idx[update, p, k] = ell_idx
                        best_amp_idx[update, p, k] = amp_idx

                        pbar.set_postfix(
                            obs_sigma=f"{float(sigma):.3g}",
                            ell=f"{float(ell):.3g}",
                            amp=f"{float(amp):.3g}",
                            K=f"{k + 1}",
                            P=f"{p + 1}",
                        )
                        pbar.update(1)

                optimal_length_scales[:, p, k] = length_scales[
                    best_ell_idx[:, p, k]
                ]
                optimal_amplitudes[:, p, k] = amplitudes[
                    best_amp_idx[:, p, k]
                ]

                # --------------------------------------------------------
                # Build F_smooth[:, p, :, k] using each neuron's best
                # partition-specific length-scale and amplitude.
                # --------------------------------------------------------
                pair_idx = np.stack(
                    [best_ell_idx[:, p, k], best_amp_idx[:, p, k]],
                    axis=1,
                )

                for ell_idx, amp_idx in np.unique(pair_idx, axis=0):
                    rows = np.flatnonzero(
                        (best_ell_idx[:, p, k] == ell_idx)
                        & (best_amp_idx[:, p, k] == amp_idx)
                    )

                    F_smooth[rows, p, :, k] = _gp_map_batch(
                        Fp_raw[rows, :, k],
                        Kmat=K_mats[int(ell_idx)][int(amp_idx)],
                        sigma=float(sigma),
                        obs_jitter=obs_jitter,
                    )

            # ------------------------------------------------------------
            # Final whole-dataset raw estimate.
            # ------------------------------------------------------------
            if equalise_final_histogram:
                idx_all, bins_all, original_counts_all, sampled_counts_all = (
                    histogram_equalised_timestep_subsample(
                        idx0=np.arange(T, dtype=np.int64),
                        bins_p=bins_p,
                        D=D,
                        rng=equalise_rng,
                        target_per_bin=target_per_bin,
                        replace=equalise_replace,
                        shuffle=True,
                    )
                )

                whole_sums, whole_counts, _, _ = _binned_partition_stats(
                    X=X,
                    S=S,
                    idx=idx_all,
                    bins=bins_all,
                    D=D,
                )

            else:
                whole_sums = np.sum(x_sums_p, axis=2)
                whole_counts = np.sum(x_counts_p, axis=2)

            F_whole_raw = np.divide(
                whole_sums,
                whole_counts,
                out=np.full_like(whole_sums, np.nan, dtype=float),
                where=whole_counts > 0,
            )

            # ------------------------------------------------------------
            # Final whole-dataset GP estimate.
            #
            # Here each neuron gets its own partition-averaged best
            # length-scale and amplitude.
            # ------------------------------------------------------------
            avg_ell_p = np.mean(optimal_length_scales[:, p, :], axis=1)
            avg_amp_p = np.mean(optimal_amplitudes[:, p, :], axis=1)

            avg_pairs = np.stack([avg_ell_p, avg_amp_p], axis=1)

            for ell, amp in np.unique(avg_pairs, axis=0):
                rows = np.flatnonzero(
                    (avg_ell_p == ell) & (avg_amp_p == amp)
                )

                Kmat = _make_gram_with_lengthscale_and_amplitude(
                    p=p,
                    ell=float(ell),
                    amplitude=float(amp),
                    domains=domains,
                    gram_fns=gram_fns,
                    gram_fn_args=gram_fn_args,
                    length_scale_key=length_scale_key,
                    amplitude_key=amplitude_key,
                )

                F_best[rows, p, :] = _gp_map_batch(
                    F_whole_raw[rows],
                    Kmat=Kmat,
                    sigma=float(sigma),
                    obs_jitter=obs_jitter,
                )

    avg_length_scales = np.mean(optimal_length_scales, axis=2)  # (N, P)
    avg_amplitudes = np.mean(optimal_amplitudes, axis=2)        # (N, P)

    # Positive means the tuning-curve model is better than the null model.
    # Negative means the tuning-curve model is worse than the null model.
    scores = np.nanmean(best_ll - null_ll, axis=2)  # (N, P)

    F_raw_mean = np.maximum(
        F_raw.mean(axis=2, keepdims=True),
        eps_rate * np.ones((N,P,1,K))
    )
    F_smooth_mean = np.maximum(
        F_smooth.mean(axis=2, keepdims=True),
        eps_rate * np.ones((N,P,1,K))
    )
    F_best_mean = np.maximum(
        F_best.mean(axis=2, keepdims=True),
        eps_rate * np.ones((N,P,1))
    )

    # F_raw_std = np.maximum(
    #     F_raw.std(axis=2, keepdims=True),
    #     eps_rate * np.ones((N,P,1,K))
    # )
    # F_smooth_std = np.maximum(
    #     F_smooth.std(axis=2, keepdims=True),
    #     eps_rate * np.ones((N,P,1,K))
    # )
    # F_best_std = np.maximum(
    #     F_best.std(axis=2, keepdims=True),
    #     eps_rate * np.ones((N,P,1))
    # )

    F_raw_normed = F_raw / F_raw_mean
    F_smooth_normed = F_smooth / F_smooth_mean
    F_best_normed = F_best / F_best_mean

    return {
        "F_raw": F_raw,
        "F_smooth": F_smooth,
        "F_best": F_best,
        "F_raw_normed": F_raw_normed,
        "F_smooth_normed": F_smooth_normed,
        "F_best_normed": F_best_normed,
        "scores": scores,
        "best_ll": best_ll,
        "null_ll": null_ll,
        "optimal_length_scales": optimal_length_scales,
        "optimal_amplitudes": optimal_amplitudes,
        "avg_length_scales": avg_length_scales,
        "avg_amplitudes": avg_amplitudes,
        "sigma": float(sigma),
        "domains": domains,
        "bin_idx": bin_idx,
    }



def _softplus_stable(x: np.ndarray):
    return np.logaddexp(0.0, x)

def angular_tuning_gram(
        sigma: float,
        D: int
):
    
    domain = np.linspace(-np.pi, np.pi, D, endpoint=False)
    delta = np.abs(domain[:, None] - domain[None, :])
    delta = np.mod(delta, 2*np.pi)
    circular_delta = np.minimum(delta, 2*np.pi - delta)

    return np.sum( np.stack(
        [np.exp(-(sigma**2 / 2) * (circular_delta + 2*np.pi*n) ** 2) for n in range(D)]
    , axis=0), axis=0)

def nonlinearity( X: np.ndarray, beta: float, b: float ): 
    # X : (N, P, D) 
    
    fr = _softplus_stable( beta * (X - b) ) 
    
    Z = np.mean(fr, axis=2, keepdims=True) 
    
    return fr / Z



def sample_tuning_curves(
    sigma: float,
    beta: float,
    b: float,
    n_samples: int,
    P: int,
    D: int,
    rng: np.random.Generator,
    jitter: float = 1e-8,
):
    """
    Sample latent GP curves and pass them through the normalised nonlinearity.

    Returns
    -------
    Q : array, shape (n_samples, P, D)
    """
    K = angular_tuning_gram(sigma=sigma, D=D)
    K = _stabilise_gram(K, jitter=jitter)

    L = np.linalg.cholesky(K)

    Z = rng.standard_normal((n_samples, P, D))
    X = np.einsum("spd,ed->spe", Z, L, optimize=True)

    Q = nonlinearity(X, beta=beta, b=b)

    return Q

def _stabilise_gram(K: np.ndarray, jitter:float) -> np.ndarray:
    D = K.shape[0]
    
    evals = np.linalg.eigvalsh(K)
    min_eval = evals[0]

    correction = jitter - min_eval

    return K + correction * np.eye(D)


def fit_sampled_tuning_curve_correlation_model(
    F: np.ndarray,                      # (N, P, D)
    sigma_values: np.ndarray,
    beta_values: np.ndarray,
    b_values: np.ndarray,
    n_samples: int = 1_000,
    fit: str = "both",                  # "global", "per_p", or "both"
    seed: int | None = None,
    jitter: float = 1e-8,
    omit_nans: bool = True,
):
    """
    Fit sampled nonlinear-GP tuning curves to empirical tuning-curve
    correlations.

    For empirical tuning curves F, compute

        C[p, d, e] = mean_n F[n, p, d] F[n, p, e].

    For sampled tuning curves Q, compute

        Qcorr[p, d, e] = mean_s Q[s, p, d] Q[s, p, e].

    Then search over sigma, beta, b to minimise

        MSE[p] = mean_{d,e} (C[p,d,e] - Qcorr[p,d,e])^2.

    The function supports either one shared parameter set across all p,
    or an independent best parameter set for each p.

    Parameters
    ----------
    F : array, shape (N, P, D)
        Empirical tuning curves, for example F_best.

    sigma_values : array
        Grid of GP covariance parameters passed to angular_tuning_gram.

    beta_values : array
        Grid of beta values for the nonlinearity.

    b_values : array
        Grid of threshold/bias values for the nonlinearity.

    n_samples : int
        Number of GP tuning curves to sample for each parameter set.

    fit : {"global", "per_p", "both"}
        Whether to fit one parameter set across all p, one independently
        for each p, or both.

    seed : int, optional
        Random seed.

    jitter : float
        Diagonal jitter added to the GP covariance before Cholesky.

    omit_nans : bool
        If True, compute empirical correlations using pairwise-valid
        non-NaN entries of F.

    Returns
    -------
    result : dict
        Contains:

        C : (P, D, D)
            Empirical correlation matrices.

        mse_per_p : (n_sigma, n_beta, n_b, P)
            MSE for each parameter setting and behavioural variable.

        mse_global : (n_sigma, n_beta, n_b)
            MSE averaged across p.

        best_global_params : dict, if requested
            Best shared parameter set.

        Q_best_global : (n_samples, P, D), if requested
            Sampled tuning curves from best shared parameter set.

        best_per_p_params : dict, if requested
            Best independent parameter values, each shape (P,).

        Q_best_per_p : (n_samples, P, D), if requested
            Sampled tuning curves where each p uses its own best parameter set.
    """
    F = np.asarray(F, dtype=float)

    if F.ndim != 3:
        raise ValueError("F must have shape (N, P, D).")

    N, P, D = F.shape

    sigma_values = np.asarray(sigma_values, dtype=float)
    beta_values = np.asarray(beta_values, dtype=float)
    b_values = np.asarray(b_values, dtype=float)

    if np.any(sigma_values <= 0):
        raise ValueError("All sigma_values must be positive.")
    if np.any(beta_values <= 0):
        raise ValueError("All beta_values must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if fit not in {"global", "per_p", "both"}:
        raise ValueError("fit must be one of {'global', 'per_p', 'both'}.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Empirical correlation:
    # C[p,d,e] = mean_n F[n,p,d] F[n,p,e]
    # ------------------------------------------------------------
    C = np.full((P, D, D), np.nan, dtype=float)

    for p in range(P):
        Fp = F[:, p, :]  # (N, D)

        if omit_nans:
            valid = np.isfinite(Fp)
            F0 = np.where(valid, Fp, 0.0)

            num = F0.T @ F0
            den = valid.astype(float).T @ valid.astype(float)

            C[p] = np.divide(
                num,
                den,
                out=np.full((D, D), np.nan, dtype=float),
                where=den > 0,
            )
        else:
            C[p] = (Fp.T @ Fp) / N

    # Fixed standard-normal samples reused across sigma values.
    # This reduces Monte Carlo noise when comparing parameter settings.
    Z = rng.standard_normal((n_samples, P, D))

    n_sigma = len(sigma_values)
    n_beta = len(beta_values)
    n_b = len(b_values)

    mse_per_p = np.full((n_sigma, n_beta, n_b, P), np.nan, dtype=float)

    # ------------------------------------------------------------
    # Grid search.
    # ------------------------------------------------------------
    total_iters = n_sigma * n_beta * n_b
    with tqdm(total=total_iters, desc="Grid search", unit="param set") as pbar:
        for i_sigma, sigma in enumerate(sigma_values):
            Kmat = angular_tuning_gram(sigma=float(sigma), D=D)
            Kmat = _stabilise_gram(Kmat, jitter=jitter)

            L = np.linalg.cholesky(Kmat)

            # X_latent[s,p,d] ~ GP(0, Kmat)
            X_latent = np.einsum("spd,ed->spe", Z, L, optimize=True)

            for i_beta, beta in enumerate(beta_values):
                for i_b, b in enumerate(b_values):
                    fr = _softplus_stable(beta * (X_latent - b))

                    # Same as nonlinearity, but numerically stable.
                    Z_norm = np.mean(fr, axis=2, keepdims=True)
                    Q = fr / Z_norm  # (n_samples, P, D)

                    Qcorr = np.einsum(
                        "spd,spe->pde",
                        Q,
                        Q,
                        optimize=True,
                    ) / n_samples

                    diff2 = (Qcorr - C) ** 2
                    mse_per_p[i_sigma, i_beta, i_b] = np.nanmean(
                        diff2,
                        axis=(1, 2),
                    )

                    pbar.set_postfix(
                        b=f"{float(b):.3g}",
                        beta=f"{float(beta):.3g}",
                        sigma=f"{float(sigma):.3g}"
                    )
                    pbar.update(1)

    mse_global = np.nanmean(mse_per_p, axis=-1)  # (n_sigma, n_beta, n_b)

    result = {
        "C": C,
        "mse_per_p": mse_per_p,
        "mse_global": mse_global,
        "sigma_values": sigma_values,
        "beta_values": beta_values,
        "b_values": b_values,
    }

    # ------------------------------------------------------------
    # Best shared parameter set across all p.
    # ------------------------------------------------------------
    if fit in {"global", "both"}:
        best_idx = np.unravel_index(np.nanargmin(mse_global), mse_global.shape)
        i_sigma, i_beta, i_b = best_idx

        best_sigma = float(sigma_values[i_sigma])
        best_beta = float(beta_values[i_beta])
        best_b = float(b_values[i_b])

        Q_best_global = sample_tuning_curves(
            sigma=best_sigma,
            beta=best_beta,
            b=best_b,
            n_samples=n_samples,
            P=P,
            D=D,
            rng=rng,
            jitter=jitter,
        )

        result["best_global_params"] = {
            "sigma": best_sigma,
            "beta": best_beta,
            "b": best_b,
            "mse": float(mse_global[best_idx]),
            "index": best_idx,
        }
        result["Q_best_global"] = Q_best_global

    # ------------------------------------------------------------
    # Best independent parameter set for each p.
    # ------------------------------------------------------------
    if fit in {"per_p", "both"}:
        best_sigma_per_p = np.empty(P, dtype=float)
        best_beta_per_p = np.empty(P, dtype=float)
        best_b_per_p = np.empty(P, dtype=float)
        best_mse_per_p = np.empty(P, dtype=float)
        best_indices_per_p = np.empty((P, 3), dtype=int)

        Q_best_per_p = np.empty((n_samples, P, D), dtype=float)

        for p in range(P):
            mse_p = mse_per_p[:, :, :, p]
            best_idx = np.unravel_index(np.nanargmin(mse_p), mse_p.shape)
            i_sigma, i_beta, i_b = best_idx

            best_sigma = float(sigma_values[i_sigma])
            best_beta = float(beta_values[i_beta])
            best_b = float(b_values[i_b])

            best_sigma_per_p[p] = best_sigma
            best_beta_per_p[p] = best_beta
            best_b_per_p[p] = best_b
            best_mse_per_p[p] = float(mse_p[best_idx])
            best_indices_per_p[p] = best_idx

            Kmat = angular_tuning_gram(sigma=best_sigma, D=D)
            Kmat = _stabilise_gram(Kmat, jitter=jitter)
            L = np.linalg.cholesky(Kmat)

            Zp = rng.standard_normal((n_samples, D))
            Xp = Zp @ L.T

            fr = _softplus_stable(best_beta * (Xp - best_b))
            Q_best_per_p[:, p, :] = fr / np.mean(fr, axis=1, keepdims=True)

        result["best_per_p_params"] = {
            "sigma": best_sigma_per_p,
            "beta": best_beta_per_p,
            "b": best_b_per_p,
            "mse": best_mse_per_p,
            "index": best_indices_per_p,
        }
        result["Q_best_per_p"] = Q_best_per_p

    return result


# === Analysis helpers (reused by steps 3-9) ===
# Depend on functions defined above: gp_cross_validated_tuning_curves,
# periodic_gaussian_gram and the `_`-helpers (GP smoothing/selection cell);
# sample_tuning_curves / fit_sampled_tuning_curve_correlation_model (modelling).


def gather_tuning_basis(F, bin_idx):
    """Evaluate tuning curves along the behavioural trajectory.

    F : (M, P, D)   tuning curves (M = N real neurons, or S sampled curves).
    bin_idx : (P, T)  behavioural bin index per timestep (-1 = invalid).
    Returns f : (M, P, T) with f[m,p,t] = F[m,p,bin_idx[p,t]], NaN where bin<0.
    """
    F = np.asarray(F, float); bin_idx = np.asarray(bin_idx)
    M, P, D = F.shape
    f = np.empty((M, P, bin_idx.shape[1]))
    for p in range(P):
        idx = bin_idx[p]; valid = idx >= 0
        g = F[:, p, :][:, np.where(valid, idx, 0)]   # (M, T)
        g[:, ~valid] = np.nan
        f[:, p, :] = g
    return f


def gather_pair_basis(F_pair, bin_idx, pairs):
    """Evaluate 2-D joint tuning maps along the trajectory.

    F_pair : (N, n_pairs, D, D);  pairs : list of (p, q).
    Returns f : (N, n_pairs, T) with f[n,j,t] = F_pair[n,j,bin_p[t],bin_q[t]].
    """
    N, npr, D, _ = F_pair.shape
    out = np.empty((N, npr, bin_idx.shape[1]))
    for j, (p, q) in enumerate(pairs):
        bp, bq = bin_idx[p], bin_idx[q]
        valid = (bp >= 0) & (bq >= 0)
        g = F_pair[:, j, np.where(valid, bp, 0), np.where(valid, bq, 0)]  # (N,T)
        g[:, ~valid] = np.nan
        out[:, j, :] = g
    return out


def _impute_columns(A):
    """Replace NaN entries of A (T,Q) with their finite column means."""
    A = A.copy()
    cm = np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    bad = ~np.isfinite(A)
    A[bad] = np.take(cm, np.nonzero(bad)[1])
    return A


def ridge_ve_per_neuron(X, F_design, l2=1e-2, test_frac=0.0, split_seed=0):
    """L2 regression of each neuron onto its OWN predictors.

    X : (N, T);  F_design : (N, Q, T) (Q predictors per neuron).
    X_est[n,t] = b_n + sum_q w[n,q] F_design[n,q,t]  (unpenalised intercept).
    Returns pooled VE and per-neuron VE (N,).

    If ``test_frac`` > 0 a shared train/test split of the timepoints is drawn
    (seeded by ``split_seed``); each neuron's weights are fit on its valid
    training timepoints and the variance explained is evaluated on its disjoint
    held-out test timepoints (baseline = training mean). ``test_frac`` = 0 is the
    in-sample fit.
    """
    X = np.asarray(X, float); F_design = np.asarray(F_design, float)
    N, T = X.shape; _, Q, _ = F_design.shape
    if test_frac and test_frac > 0:
        perm = np.random.default_rng(split_seed).permutation(T)
        n_test = int(round(test_frac * T))
        is_test = np.zeros(T, bool); is_test[perm[:n_test]] = True
    else:
        is_test = np.zeros(T, bool)                       # in-sample: train == test
    ve = np.full(N, np.nan); ss_res_tot = ss_tot_tot = 0.0
    for n in range(N):
        A = F_design[n].T; y = X[n]
        valid = np.isfinite(y) & np.any(np.isfinite(A), axis=1)
        tr = valid & ~is_test if test_frac > 0 else valid
        te = valid & is_test if test_frac > 0 else valid
        if tr.sum() < Q + 2 or te.sum() < 1:
            continue
        Ac0 = _impute_columns(A[tr]); amean = Ac0.mean(0, keepdims=True)
        ytr = y[tr]; ybar = ytr.mean()
        w = np.linalg.solve((Ac0 - amean).T @ (Ac0 - amean) + l2 * np.eye(Q),
                            (Ac0 - amean).T @ (ytr - ybar))
        yhat = (_impute_columns(A[te]) - amean) @ w + ybar
        sr = float(np.sum((y[te] - yhat) ** 2)); st = float(np.sum((y[te] - ybar) ** 2))
        ss_res_tot += sr; ss_tot_tot += st
        ve[n] = 1.0 - sr / st if st > 0 else np.nan
    pooled = 1.0 - ss_res_tot / ss_tot_tot if ss_tot_tot > 0 else np.nan
    return pooled, ve


def ridge_ve_shared_basis(X, design, l2=1e-2, test_frac=0.0, split_seed=0):
    """L2 regression of EVERY neuron onto a shared predictor design.

    X : (N, T);  design : (Q, T) common to all neurons (e.g. Q = S*P sampled
    curves). Solved jointly for all neurons. Returns pooled VE and per-neuron VE.

    If ``test_frac`` > 0 the valid timepoints are randomly split (seeded by
    ``split_seed``): the ridge weights are fit on the training fraction and the
    variance explained is evaluated on the disjoint held-out test fraction
    (baseline = training mean). ``test_frac`` = 0 reproduces the in-sample fit.
    """
    X = np.asarray(X, float); design = np.asarray(design, float)
    A = design.T
    rows = np.all(np.isfinite(A), axis=1)
    A = A[rows]; Xv = X[:, rows]
    Tr, Q = A.shape
    if test_frac and test_frac > 0:
        perm = np.random.default_rng(split_seed).permutation(Tr)
        n_test = int(round(test_frac * Tr))
        test_i, train_i = perm[:n_test], perm[n_test:]
    else:
        test_i = train_i = np.arange(Tr)                  # in-sample
    A_tr, A_te = A[train_i], A[test_i]
    X_tr, X_te = Xv[:, train_i], Xv[:, test_i]
    amean = A_tr.mean(0, keepdims=True)
    Ac_tr = A_tr - amean
    Ginv_At = np.linalg.solve(Ac_tr.T @ Ac_tr + l2 * np.eye(Q), Ac_tr.T)   # (Q, Ttr)
    xbar = np.nanmean(X_tr, axis=1, keepdims=True)         # train baseline
    W = np.nan_to_num(X_tr - xbar) @ Ginv_At.T            # (N, Q)
    Xhat = W @ (A_te - amean).T + xbar                     # predict held-out
    ss_res = np.nansum((X_te - Xhat) ** 2, axis=1)
    ss_tot = np.nansum((X_te - xbar) ** 2, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ve = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    pooled = 1.0 - ss_res.sum() / ss_tot.sum() if ss_tot.sum() > 0 else np.nan
    return pooled, ve


def sample_basis_per_p(sigma, beta, b, S, D, rng, jitter=1e-6):
    """Sample S tuning curves per behavioural variable from the fitted
    generative process (per-variable params). Returns (S, P, D)."""
    P = len(sigma)
    out = np.empty((S, P, D))
    for p in range(P):
        Q = sample_tuning_curves(sigma=float(sigma[p]), beta=float(beta[p]),
                                 b=float(b[p]), n_samples=S, P=1, D=D, rng=rng,
                                 jitter=jitter)
        out[:, p, :] = Q[:, 0, :]
    return out


def _gp_map_shared_obs(Y, Kmat, sigma, obs_mask, obs_jitter=0.0):
    """GP posterior MAP for rows of Y when the observed grid cells are shared
    across all rows (true here: a cell is observed iff its dwell-time > 0).
    Solves once for all rows. Y : (N, G) with NaN at unobserved cells."""
    N, G = Y.shape
    out = np.zeros((N, G)); obs = np.asarray(obs_mask, bool)
    if not obs.any():
        return out
    R = Kmat[np.ix_(obs, obs)] + (sigma ** 2 + obs_jitter) * np.eye(obs.sum())
    Z = np.linalg.solve(R, np.nan_to_num(Y[:, obs]).T)   # (n_obs, N)
    return (Kmat[:, obs] @ Z).T


def gp_cross_validated_tuning_curves_2d(
    X, B, S, D=20, K=2, L=2, length_scales=None, amplitudes=None, sigma=1.0,
    gram_fn_args=None, dt=0.025, obs_jitter=0.0, eps_rate=1e-12, pairs=None,
):
    """Cross-validated GP tuning on 1-D singletons AND 2-D variable pairs.

    Returns dict: F_best_single (N,P,D), scores_single (N,P),
    F_best_pair (N,n_pairs,D,D), scores_pair (N,n_pairs), pairs, bin_idx, domains.
    """
    X = np.asarray(X, float); B = np.asarray(B, float); S = np.asarray(S, float)
    N, T = X.shape; P = B.shape[0]
    if length_scales is None: length_scales = np.logspace(-2, 0, 8)
    if amplitudes is None: amplitudes = np.logspace(-2, 0, 8)
    length_scales = np.asarray(length_scales, float); amplitudes = np.asarray(amplitudes, float)
    if gram_fn_args is None: gram_fn_args = {"period": 2 * np.pi}
    period = gram_fn_args.get("period")

    # 1-D singletons via the validated routine
    single = gp_cross_validated_tuning_curves(
        X, B, S, D=D, K=K, L=L, length_scales=length_scales, amplitudes=amplitudes,
        sigma=sigma, gram_fns=periodic_gaussian_gram, gram_fn_args=dict(gram_fn_args),
        length_scale_key="length_scale", dt=dt, obs_jitter=obs_jitter,
        equalise_histogram=False)
    bin_idx = single['bin_idx']; domains = single['domains']

    partitions = _partition_indices(T, K, L)
    pairs = list(combinations(range(P), 2)) if pairs is None else list(pairs)
    n_pairs = len(pairs); D2 = D * D
    F_best_pair = np.full((N, n_pairs, D, D), np.nan)
    scores_pair = np.full((N, n_pairs), np.nan)

    for pi, (p, q) in enumerate(tqdm(pairs, desc="2D pairs", unit="pair")):
        bp, bq = bin_idx[p], bin_idx[q]
        jb = np.where((bp >= 0) & (bq >= 0), bp * D + bq, -1)
        x_sums = np.zeros((N, D2, K)); x_counts = np.zeros((N, D2, K))
        spk = np.zeros((N, D2, K)); dwell = np.zeros((D2, K))
        for k, idx0 in enumerate(partitions):
            v = jb[idx0] >= 0; idx = idx0[v]; b = jb[idx0][v]
            xs, xc, scc, dw = _binned_partition_stats(X=X, S=S, idx=idx, bins=b, D=D2)
            x_sums[:, :, k] = xs; x_counts[:, :, k] = xc
            spk[:, :, k] = scc; dwell[:, k] = dw
        Fp_raw = np.divide(x_sums, x_counts, out=np.full_like(x_sums, np.nan),
                           where=x_counts > 0)
        # cache Kronecker grams (unit-amplitude axes, scaled by amp^2)
        Ka = {ie: periodic_gaussian_gram(domains[p], length_scale=float(ell),
              amplitude=1.0, period=period) for ie, ell in enumerate(length_scales)}
        Kb = {ie: periodic_gaussian_gram(domains[q], length_scale=float(ell),
              amplitude=1.0, period=period) for ie, ell in enumerate(length_scales)}
        Kmats = {(ie, ia): (amp ** 2) * np.kron(Ka[ie], Kb[ie])
                 for ie in range(len(length_scales))
                 for ia, amp in enumerate(amplitudes)}
        best_ll = np.full((N, K), -np.inf); null_ll = np.full((N, K), np.nan)
        best_ie = np.zeros((N, K), int); best_ia = np.zeros((N, K), int)
        for k in range(K):
            Y_train = Fp_raw[:, :, k]; obs_k = dwell[:, k] > 0
            null_ll[:, k] = _crossvalidated_null_poisson_ll(
                lambda_null=np.nanmean(Y_train, axis=1), spike_counts_p=spk,
                dwell_counts_p=dwell, k=k, dt=dt)
            for ie in range(len(length_scales)):
                for ia in range(len(amplitudes)):
                    rates = np.clip(_gp_map_shared_obs(Y_train, Kmats[(ie, ia)],
                                    float(sigma), obs_k, obs_jitter), eps_rate, None)
                    lr = np.log(rates); ll = np.zeros(N)
                    for r in range(K):
                        if r == k: continue
                        ll += (np.einsum("nd,nd->n", spk[:, :, r], lr)
                               - dt * np.einsum("d,nd->n", dwell[:, r], rates))
                    ll /= K - 1; upd = ll > best_ll[:, k]
                    best_ll[upd, k] = ll[upd]; best_ie[upd, k] = ie; best_ia[upd, k] = ia
        avg_ell = length_scales[best_ie].mean(1); avg_amp = amplitudes[best_ia].mean(1)
        wc = x_counts.sum(2)
        whole = np.divide(x_sums.sum(2), wc, out=np.full((N, D2), np.nan), where=wc > 0)
        obs_all = dwell.sum(1) > 0
        for ell, amp in np.unique(np.stack([avg_ell, avg_amp], 1), axis=0):
            rows = np.flatnonzero((avg_ell == ell) & (avg_amp == amp))
            Kn = (amp ** 2) * np.kron(
                periodic_gaussian_gram(domains[p], length_scale=float(ell), amplitude=1.0, period=period),
                periodic_gaussian_gram(domains[q], length_scale=float(ell), amplitude=1.0, period=period))
            fm = _gp_map_shared_obs(whole[rows], Kn, float(sigma), obs_all, obs_jitter)
            F_best_pair[rows, pi] = fm.reshape(len(rows), D, D)
        scores_pair[:, pi] = np.nanmean(best_ll - null_ll, axis=1)

    return dict(F_best_single=single['F_best'], scores_single=single['scores'],
                F_best_pair=F_best_pair, scores_pair=scores_pair, pairs=pairs,
                bin_idx=bin_idx, domains=domains)


def circular_com(F, domain):
    """Circular center of mass of tuning curves over an angular domain.
    F : (N, P, D) non-negative tuning curves; domain : length-P list of (D,).
    Returns (N, P) angles in radians (NaN where a curve has no mass)."""
    F = np.clip(np.nan_to_num(F, nan=0.0), 0, None)
    N, P, D = F.shape
    com = np.full((N, P), np.nan)
    for p in range(P):
        x = np.asarray(domain[p]); w = F[:, p, :]
        ok = w.sum(1) > 0
        com[ok, p] = np.arctan2((w * np.sin(x)).sum(1)[ok], (w * np.cos(x)).sum(1)[ok])
    return com


def circular_corr(a, b):
    """Circular correlation coefficient between two angle arrays (NaNs dropped)."""
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    abar = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    bbar = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - abar), np.sin(b - bbar)
    denom = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / denom) if denom > 0 else np.nan


# ============================================================================
# Presentation-specific helpers
# ============================================================================

def significance_threshold(n_global, alpha=0.05):
    """Score threshold T = -log10(alpha / n_global)."""
    return float(-np.log10(alpha / n_global))


def significance_mask(scores, T):
    """Boolean mask over neurons: kept if max over variables of score > T.
    scores : (N, P) mean held-out delta-LL per (neuron, variable)."""
    return np.nanmax(scores, axis=1) > T


def _circular_com_index(curve, domain):
    """Circular centre-of-mass of a 1-D curve, returned as a fractional bin
    index in [0, D). `curve` is clipped at 0 to act as non-negative mass."""
    w = np.clip(np.nan_to_num(curve, nan=0.0), 0, None)
    D = w.shape[-1]
    ang = np.linspace(-np.pi, np.pi, D, endpoint=False)
    if w.sum() <= 0:
        return D / 2.0
    com_ang = np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())
    return (com_ang + np.pi) / (2 * np.pi) * D


def align_by_com_1d(curves, center=True):
    """Circularly roll each (N, D) curve so its circular COM sits at the centre.
    Returns the aligned copy. NaNs are treated as zero mass for the COM."""
    curves = np.asarray(curves, float)
    N, D = curves.shape
    out = np.full_like(curves, np.nan)
    target = D // 2 if center else 0
    for n in range(N):
        com = _circular_com_index(curves[n], None)
        shift = int(np.round(target - com))
        out[n] = np.roll(curves[n], shift)
    return out


def align_by_com_2d(maps, center=True):
    """Circularly roll each (N, D, D) map so its 2-D circular COM is centred
    along both (periodic) axes."""
    maps = np.asarray(maps, float)
    N, D, _ = maps.shape
    out = np.full_like(maps, np.nan)
    target = D // 2 if center else 0
    for n in range(N):
        m = np.clip(np.nan_to_num(maps[n], nan=0.0), 0, None)
        com_r = _circular_com_index(m.sum(1), None)   # marginal over axis 1
        com_c = _circular_com_index(m.sum(0), None)   # marginal over axis 0
        out[n] = np.roll(np.roll(maps[n], int(np.round(target - com_r)), axis=0),
                         int(np.round(target - com_c)), axis=1)
    return out


def tuning_correlation(curves):
    """Empirical correlation matrix C[d,e] = mean_n curves[n,d] curves[n,e]
    (the definition used in the network-fitting / generative section).
    NaN-robust: averages over neurons with finite entries for each (d,e)."""
    curves = np.asarray(curves, float)
    valid = np.isfinite(curves)
    F0 = np.where(valid, curves, 0.0)
    num = F0.T @ F0
    den = valid.astype(float).T @ valid.astype(float)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def mean_subtracted_svals(C, k=None):
    """Singular values of the mean-subtracted matrix ``C - mean(C)``.

    C : (D, D) (e.g. a tuning correlation matrix). The scalar mean is removed
    before the SVD so the constant (DC) component does not dominate the
    spectrum. Returns the first ``k`` singular values (all if ``k`` is None)."""
    C = np.nan_to_num(np.asarray(C, float))
    C = C - C.mean()
    s = np.linalg.svd(C, compute_uv=False)
    return s if k is None else s[:k]


def onehot_bin_design(bin_idx_row, D):
    """One-hot design (T, D) for a single binned variable; unvisited/invalid
    timesteps (bin < 0) get an all-zero row."""
    T = bin_idx_row.shape[0]
    M = np.zeros((T, D))
    ok = bin_idx_row >= 0
    M[np.nonzero(ok)[0], bin_idx_row[ok]] = 1.0
    return M


def onehot_joint_design(bp, bq, D):
    """One-hot design (T, D*D) for a pair of binned variables (joint bin)."""
    T = bp.shape[0]
    M = np.zeros((T, D * D))
    ok = (bp >= 0) & (bq >= 0)
    idx = np.nonzero(ok)[0]
    M[idx, bp[ok] * D + bq[ok]] = 1.0
    return M


# ===========================================================================
# D-dimensional ('torus') generalisation of the generative tuning model
# ===========================================================================


# ---------------------------------------------------------------------------
# Empirical D-dimensional tuning curves on a periodic B^D grid
# ---------------------------------------------------------------------------
def bin_behaviour(B_behav, n_bins):
    """Bin each behavioural variable into ``n_bins`` equal-width bins.

    B_behav : (P, T) behavioural variables.
    Returns bin_idx (P, T) integer bins in {0,...,n_bins-1} (-1 = invalid) and a
    length-P list of bin-centre domains.
    """
    P, T = B_behav.shape
    bin_idx = np.full((P, T), -1, dtype=int)
    domains = []
    for p in range(P):
        b, dom = _bin_behaviour_row_and_domain(B_behav[p], n_bins)
        bin_idx[p] = b
        domains.append(dom)
    return bin_idx, domains


def empirical_torus_tuning(X, bin_idx, var_tuples, n_bins):
    """Raw binned tuning curves on a periodic B^D grid for each variable tuple.

    X          : (N, T) neural activity (firing rates).
    bin_idx    : (P, T) per-variable bin index (-1 = invalid), from bin_behaviour.
    var_tuples : list of D-tuples of variable indices (the 'tuning bases').
    n_bins     : bins per dimension B (so each tuning lives on M = B**D cells).

    Returns
    -------
    F      : (N, n_bases, M)  mean firing rate per joint cell (NaN where unvisited).
    counts : (n_bases, M)     occupancy (number of timesteps) per joint cell.
    """
    X = np.asarray(X, float)
    N, T = X.shape
    n_bases = len(var_tuples)
    D = len(var_tuples[0])
    M = n_bins ** D
    F = np.full((N, n_bases, M), np.nan)
    counts = np.zeros((n_bases, M), dtype=int)
    for j, vt in enumerate(var_tuples):
        assert len(vt) == D, "all tuples must have the same dimension D"
        # joint flattened cell index = sum_k bin[v_k] * B**(D-1-k)
        valid = np.all([bin_idx[v] >= 0 for v in vt], axis=0)
        cell = np.zeros(T, dtype=int)
        for k, v in enumerate(vt):
            cell += np.where(bin_idx[v] >= 0, bin_idx[v], 0) * (n_bins ** (D - 1 - k))
        cell = np.where(valid, cell, -1)
        ok = cell >= 0
        counts[j] = np.bincount(cell[ok], minlength=M)
        # per-neuron sum of firing rate in each cell, then divide by count
        sums = np.zeros((N, M))
        np.add.at(sums.T, cell[ok], X[:, ok].T)
        with np.errstate(invalid="ignore"):
            F[:, j, :] = np.where(counts[j] > 0, sums / counts[j], np.nan)
    return F, counts


# ---------------------------------------------------------------------------
# D-dimensional periodic GP currents (Kronecker structure)
# ---------------------------------------------------------------------------
def _torus_cholesky(sigma, n_bins, jitter=1e-6):
    """Cholesky factor of the 1-D angular GP gram. The D-dim torus gram is the
    Kronecker product of D copies, so its factor is L1 (x) ... (x) L1."""
    K1 = _stabilise_gram(angular_tuning_gram(sigma, n_bins), jitter=jitter)
    return np.linalg.cholesky(K1)


def _apply_torus_factor(z, L1, D):
    """Apply (L1 (x) ... (x) L1) to z whose last D axes index the grid.
    Equivalent to multiplying by the Kronecker Cholesky factor without forming it."""
    x = z
    nd = x.ndim
    for axis in range(nd - D, nd):
        x = np.moveaxis(x, axis, -1) @ L1.T
        x = np.moveaxis(x, -1, axis)
    return x


def sample_torus_tuning_curves(sigma, beta, b, n_samples, n_bases, n_bins, D,
                               rng, jitter=1e-6, z=None):
    """Sample latent GP currents on a (B,)*D torus and pass them through the
    normalised soft-plus. Returns Q of shape (n_samples, n_bases, B**D)."""
    L1 = _torus_cholesky(sigma, n_bins, jitter)
    if z is None:
        z = rng.standard_normal((n_samples, n_bases) + (n_bins,) * D)
    x = _apply_torus_factor(z, L1, D).reshape(n_samples, n_bases, n_bins ** D)
    fr = _softplus_stable(beta * (x - b))
    return fr / fr.mean(axis=2, keepdims=True)


def _corr_omit_nan(curves):
    """Correlation matrix mean_n curves[n,d] curves[n,e], NaN-robust over n."""
    valid = np.isfinite(curves)
    C0 = np.where(valid, curves, 0.0)
    num = C0.T @ C0
    den = valid.astype(float).T @ valid.astype(float)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


# ---------------------------------------------------------------------------
# Fit the D-dimensional generative model to empirical tuning correlations
# ---------------------------------------------------------------------------
def fit_torus_generative_model(
    F, bin_counts, n_bins, D,
    sigma_values, beta_values, b_values,
    min_count=1, n_samples=1000, fit="per_p", seed=None, jitter=1e-6,
    renormalise=True, return_corr=True,
):
    """Fit the GP -> soft-plus generative model of tuning curves in D dimensions.

    The tuned 'currents' are drawn from a periodic Gaussian process on a
    D-dimensional torus (an isotropic kernel = Kronecker product of D identical
    1-D angular kernels with covariance parameter ``sigma``) and passed through
    the normalised soft-plus ``f(x) = softplus(beta (x - b)) / Z``. For each
    'tuning basis' the hyperparameters (sigma, beta, b) are chosen to minimise
    the MSE between the empirical and sampled tuning-correlation tensors, masked
    to cells whose occupancy ``bin_counts`` is at least ``min_count``.

    With D=1 this reproduces ``fit_sampled_tuning_curve_correlation_model``.

    Parameters
    ----------
    F          : (N, n_bases, M)  empirical tuning curves, M = n_bins**D.
    bin_counts : (n_bases, M)     occupancy per cell (for the MSE mask).
    n_bins, D  : bins per dimension and the dimensionality.
    sigma_values, beta_values, b_values : grid-search axes.
    min_count  : a correlation entry (c1,c2) enters the MSE only if both cells
                 have occupancy >= min_count.
    renormalise: rescale each empirical/sampled curve to unit mean over the
                 retained cells before correlating (scale-matches the two).

    Returns dict with: mse (n_sigma,n_beta,n_b,n_bases), best_params
    (sigma/beta/b/mse/index, each length n_bases), valid_masks (n_bases,M),
    sigma_values/beta_values/b_values; and if ``return_corr``: C and Qcorr_best
    as (n_bases, M, M) tensors (NaN outside the valid block).
    """
    F = np.asarray(F, float)
    N, n_bases, M = F.shape
    assert M == n_bins ** D, "F last axis must equal n_bins**D"
    bin_counts = np.asarray(bin_counts)
    rng = np.random.default_rng(seed)

    valid_masks = bin_counts >= min_count               # (n_bases, M)
    valid_idx = [np.flatnonzero(valid_masks[j]) for j in range(n_bases)]

    # Empirical correlations on the retained cells (per basis).
    C_valid = []
    for j in range(n_bases):
        Fv = F[:, j, valid_idx[j]]
        if renormalise:
            Fv = Fv / np.nanmean(Fv, axis=1, keepdims=True)
        C_valid.append(_corr_omit_nan(Fv))

    n_sigma, n_beta, n_b = len(sigma_values), len(beta_values), len(b_values)
    mse = np.full((n_sigma, n_beta, n_b, n_bases), np.nan)

    # Fixed standard normals reused across the grid (variance reduction).
    z = rng.standard_normal((n_samples, n_bases) + (n_bins,) * D)

    Qcorr_best = [None] * n_bases
    best_mse = np.full(n_bases, np.inf)

    with tqdm(total=n_sigma * n_beta * n_b, desc=f"D={D} grid search",
              unit="param") as pbar:
        for i_s, sigma in enumerate(sigma_values):
            L1 = _torus_cholesky(float(sigma), n_bins, jitter)
            x = _apply_torus_factor(z, L1, D).reshape(n_samples, n_bases, M)
            xv = [x[:, j, valid_idx[j]] for j in range(n_bases)]   # restrict to valid
            for i_be, beta in enumerate(beta_values):
                for i_b, b in enumerate(b_values):
                    for j in range(n_bases):
                        fr = _softplus_stable(beta * (xv[j] - b))
                        Qv = fr / fr.mean(axis=1, keepdims=True)
                        Qcorr = (Qv.T @ Qv) / n_samples
                        m = float(np.nanmean((C_valid[j] - Qcorr) ** 2))
                        mse[i_s, i_be, i_b, j] = m
                        if m < best_mse[j]:
                            best_mse[j] = m
                            Qcorr_best[j] = Qcorr
                    pbar.update(1)
                    pbar.set_postfix(sigma=f"{float(sigma):.2g}",
                                     beta=f"{float(beta):.2g}", b=f"{float(b):.2g}")

    # Best parameters per basis.
    best = dict(sigma=np.empty(n_bases), beta=np.empty(n_bases),
                b=np.empty(n_bases), mse=np.empty(n_bases),
                index=np.empty((n_bases, 3), int))
    for j in range(n_bases):
        idx = np.unravel_index(np.nanargmin(mse[..., j]), mse[..., j].shape)
        best["sigma"][j] = sigma_values[idx[0]]
        best["beta"][j] = beta_values[idx[1]]
        best["b"][j] = b_values[idx[2]]
        best["mse"][j] = mse[idx + (j,)]
        best["index"][j] = idx

    result = dict(mse=mse, best_params=best, valid_masks=valid_masks,
                  sigma_values=np.asarray(sigma_values),
                  beta_values=np.asarray(beta_values),
                  b_values=np.asarray(b_values))

    if return_corr:
        C_full = np.full((n_bases, M, M), np.nan)
        Q_full = np.full((n_bases, M, M), np.nan)
        for j in range(n_bases):
            ix = valid_idx[j]
            C_full[j][np.ix_(ix, ix)] = C_valid[j]
            if Qcorr_best[j] is not None:
                Q_full[j][np.ix_(ix, ix)] = Qcorr_best[j]
        result["C"] = C_full
        result["Qcorr_best"] = Q_full
    return result


# ===========================================================================
# D-dimensional cross-validated GP tuning (generalises the 2-D routine)
# ===========================================================================


def _kron_subset_gram(grams_1d, occ_idx):
    """K_oo[a,b] = prod_d grams_1d[d][i_d(a), i_d(b)] for occupied cells.
    grams_1d : list of D (B,B) matrices; occ_idx : tuple of D arrays (n_occ,)."""
    K = np.ones((occ_idx[0].size, occ_idx[0].size))
    for g, ix in zip(grams_1d, occ_idx):
        K = K * g[np.ix_(ix, ix)]
    return K


def _gp_map_occ(Y, K_oo, sigma, obs, obs_jitter=0.0):
    """Posterior MAP at the occupied cells. Y : (N, n_occ) with NaN where not
    observed in this partition; obs : (n_occ,) bool for the training cells."""
    N, n_occ = Y.shape
    out = np.zeros((N, n_occ))
    if not obs.any():
        return out
    R = K_oo[np.ix_(obs, obs)] + (sigma ** 2 + obs_jitter) * np.eye(obs.sum())
    Z = np.linalg.solve(R, np.nan_to_num(Y[:, obs]).T)   # (n_obs, N)
    return (K_oo[:, obs] @ Z).T


def gp_cross_validated_tuning_curves_nd(
    X, B_behav, S, var_tuples, n_bins=10, K=2, L_part=2,
    length_scales=None, amplitudes=None, sigma=1.0,
    period=2 * np.pi, dt=0.025, obs_jitter=0.0, eps_rate=1e-12,
):
    """Cross-validated GP tuning on a periodic ``n_bins**D`` grid for each tuple
    of behavioural variables. Generalises gp_cross_validated_tuning_curves_2d to
    arbitrary D (D = len(var_tuples[0])); D=1 gives ordinary 1-D tuning curves.

    Returns dict: F_best (N, n_bases, M) tuning (NaN at unvisited cells, M=B**D),
    counts (n_bases, M) whole-session occupancy, scores (N, n_bases),
    bin_idx (P, T), domains, var_tuples, n_bins.
    """
    X = np.asarray(X, float); B_behav = np.asarray(B_behav, float); S = np.asarray(S, float)
    N, T = X.shape
    D = len(var_tuples[0])
    M = n_bins ** D
    if length_scales is None: length_scales = np.logspace(-2, 0, 8)
    if amplitudes is None: amplitudes = np.logspace(-2, 0, 8)
    length_scales = np.asarray(length_scales, float); amplitudes = np.asarray(amplitudes, float)

    bin_idx, domains = bin_behaviour(B_behav, n_bins)
    partitions = _partition_indices(T, K, L_part)

    n_bases = len(var_tuples)
    F_best = np.full((N, n_bases, M), np.nan)
    counts_out = np.zeros((n_bases, M), int)
    scores = np.full((N, n_bases), np.nan)

    from tqdm.auto import tqdm
    for j, vt in enumerate(tqdm(var_tuples, desc=f"D={D} CV tuning", unit="basis")):
        # joint flattened cell index per timestep
        valid = np.all([bin_idx[v] >= 0 for v in vt], axis=0)
        cell = np.zeros(T, int)
        for k, v in enumerate(vt):
            cell += np.where(bin_idx[v] >= 0, bin_idx[v], 0) * (n_bins ** (D - 1 - k))
        cell = np.where(valid, cell, -1)

        # occupied cells -> compact occ-space
        total = np.bincount(cell[cell >= 0], minlength=M)
        counts_out[j] = total
        occ_flat = np.flatnonzero(total > 0)
        n_occ = occ_flat.size
        occ_pos = np.full(M, -1, int); occ_pos[occ_flat] = np.arange(n_occ)
        cell_occ = np.where(cell >= 0, occ_pos[cell], -1)
        occ_idx = np.unravel_index(occ_flat, (n_bins,) * D)   # per-dim indices

        # per-partition binned stats in occ-space
        x_sums = np.zeros((N, n_occ, K)); x_counts = np.zeros((N, n_occ, K))
        spk = np.zeros((N, n_occ, K)); dwell = np.zeros((n_occ, K))
        for kk, idx0 in enumerate(partitions):
            v = cell_occ[idx0] >= 0; idx = idx0[v]; b = cell_occ[idx0][v]
            xs, xc, sc, dw = _binned_partition_stats(X=X, S=S, idx=idx, bins=b, D=n_occ)
            x_sums[:, :, kk] = xs; x_counts[:, :, kk] = xc
            spk[:, :, kk] = sc; dwell[:, kk] = dw
        Fp_raw = np.divide(x_sums, x_counts, out=np.full_like(x_sums, np.nan),
                           where=x_counts > 0)

        # 1-D grams per dimension, per length-scale (amplitude applied as amp**2)
        grams = {ie: [periodic_gaussian_gram(domains[v], length_scale=float(ell),
                      amplitude=1.0, period=period) for v in vt]
                 for ie, ell in enumerate(length_scales)}
        Koo = {(ie, ia): (amp ** 2) * _kron_subset_gram(grams[ie], occ_idx)
               for ie in range(len(length_scales)) for ia, amp in enumerate(amplitudes)}

        best_ll = np.full((N, K), -np.inf); null_ll = np.full((N, K), np.nan)
        best_ie = np.zeros((N, K), int); best_ia = np.zeros((N, K), int)
        for kk in range(K):
            Y_train = Fp_raw[:, :, kk]; obs_k = dwell[:, kk] > 0
            null_ll[:, kk] = _crossvalidated_null_poisson_ll(
                lambda_null=np.nanmean(Y_train, axis=1), spike_counts_p=spk,
                dwell_counts_p=dwell, k=kk, dt=dt)
            for ie in range(len(length_scales)):
                for ia in range(len(amplitudes)):
                    rates = np.clip(_gp_map_occ(Y_train, Koo[(ie, ia)], float(sigma),
                                                obs_k, obs_jitter), eps_rate, None)
                    lr = np.log(rates); ll = np.zeros(N)
                    for r in range(K):
                        if r == kk: continue
                        ll += (np.einsum("nd,nd->n", spk[:, :, r], lr)
                               - dt * np.einsum("d,nd->n", dwell[:, r], rates))
                    ll /= K - 1; upd = ll > best_ll[:, kk]
                    best_ll[upd, kk] = ll[upd]; best_ie[upd, kk] = ie; best_ia[upd, kk] = ia

        avg_ell = length_scales[best_ie].mean(1); avg_amp = amplitudes[best_ia].mean(1)
        wc = x_counts.sum(2)
        whole = np.divide(x_sums.sum(2), wc, out=np.full((N, n_occ), np.nan), where=wc > 0)
        obs_all = dwell.sum(1) > 0
        for ell, amp in np.unique(np.stack([avg_ell, avg_amp], 1), axis=0):
            rows = np.flatnonzero((avg_ell == ell) & (avg_amp == amp))
            g1 = [periodic_gaussian_gram(domains[v], length_scale=float(ell),
                  amplitude=1.0, period=period) for v in vt]
            Kn = (amp ** 2) * _kron_subset_gram(g1, occ_idx)
            fm = _gp_map_occ(whole[rows], Kn, float(sigma), obs_all, obs_jitter)
            F_best[np.ix_(rows, [j], occ_flat)] = fm[:, None, :]
        scores[:, j] = np.nanmean(best_ll - null_ll, axis=1)

    return dict(F_best=F_best, counts=counts_out, scores=scores,
                bin_idx=bin_idx, domains=domains, var_tuples=var_tuples, n_bins=n_bins)
