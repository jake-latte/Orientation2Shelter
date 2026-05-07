import numpy as np
import os
import polars as pl
import json
from typing import Any, Mapping
from pathlib import Path
import sys

base_path = '/ceph/branco/'
sub_path = 'Jasmine_Laurence/Experimental_Data'
save_path = '/ceph/branco/Jake/training_data'

fps = 40
frame_time = 1 / fps
window_time = 0.1
window_size = int(window_time // frame_time) 



###########################################################################################
#                                          HELPERS                                        #
###########################################################################################



def serialize_metadata(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(',', ':'))

def write_csv(df: pl.DataFrame, path: str):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.write_csv(
        path, 
        include_header=True
    )









###########################################################################################
#                                GET BEHAVIOURAL DATA                                     #
###########################################################################################

def get_behavioural_data(animal_idx, exp_idx, animal, exp_path, dt=0.025, min_dist=200, 
                         length_scale: float=5.0, truncate:float=3.0, mode='edge'):

    def _squared_exp_kernel_1d() -> np.ndarray:
        """
        Construct a normalized squared-exponential / Gaussian kernel.

        k[t] ∝ exp(-t^2 / (2 sigma_frames^2))

        length_scale is the kernel length-scale in units of frames.
        """
        if length_scale is None or length_scale <= 0:
            return np.array([1.0])

        radius = int(np.ceil(truncate * length_scale))
        offsets = np.arange(-radius, radius + 1)

        kernel = np.exp(
            -0.5 * (offsets / length_scale) ** 2
        )

        kernel = kernel / kernel.sum()

        return kernel


    def _squared_exp_smooth_1d(
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Smooth a 1D time-series by convolving with a normalized
        squared-exponential kernel.
        """
        z = np.asarray(z, dtype=float)

        if length_scale is None or length_scale <= 0:
            return z.copy()

        kernel = _squared_exp_kernel_1d()

        radius = len(kernel) // 2

        z_pad = np.pad(
            z,
            pad_width=radius,
            mode=mode,
        )

        z_smooth = np.convolve(
            z_pad,
            kernel,
            mode="valid",
        )

        return z_smooth


    def get_angular_speed(
        hd: pl.Series
    ) -> pl.Series:
        """
        Smooth the unwrapped heading direction using a squared-exponential
        convolution kernel, then compute angular speed.

        hd: heading direction in radians.
        """
        hd_ = hd.to_numpy().astype(float)

        # Unwrap circular angle before smoothing
        hd_unwrapped = np.unwrap(hd_)

        # Smooth unwrapped heading direction
        hd_smooth = _squared_exp_smooth_1d(hd_unwrapped)

        d_theta = np.diff(hd_smooth)

        angular_speed = np.concatenate(([0.0], d_theta)) / dt

        return pl.Series(
            name="angular_speed",
            values=angular_speed,
        )

    def get_translational_speed(
        x: pl.Series,
        y: pl.Series
    ) -> pl.Series:
        """
        Smooth x(t) and y(t) using a squared-exponential convolution kernel,
        then compute frame-to-frame translational speed.
        """
        x_ = x.to_numpy().astype(float)
        y_ = y.to_numpy().astype(float)

        x_smooth = _squared_exp_smooth_1d(x_)

        y_smooth = _squared_exp_smooth_1d(y_)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)

        d = np.sqrt(dx**2 + dy**2)

        translational_speed = np.concatenate(([0.0], d)) / dt

        return pl.Series(
            name="translational_speed",
            values=translational_speed,
        )

    full_path = os.path.join(base_path, sub_path, animal, exp_path, "processed_data", "full_video_dataframe.csv")
    session_df = (
        pl.scan_csv(full_path)
        .select(['frames', 'hdir', 'hsa', 'mouse_x_position', 'mouse_y_position',
                'homingPeriod', 'OutofshelterIdx', 'EscapePeriod', 'shelter', 'barrier_present'])
        .with_columns(
            (
                pl.lit(animal_idx).alias("animal_idx"), 
                pl.lit(exp_idx).alias("exp_idx")
            )
        )
        .rename({
            "mouse_x_position": "x",
            "mouse_y_position": "y",
            "frames": "frame",
        })
        .with_columns(
            (
                ((pl.col('x') - 512)**2 + (pl.col('y') - 1024)**2)**0.5
            ).alias('dist')
        )
        .with_columns(
            (
                pl.col('dist') >= min_dist*10
            ).alias('far_from_shelter')
        )
        .collect()
    )

    session_df = (
        session_df
        .with_columns(
            get_angular_speed(session_df['hdir']),
            get_translational_speed(session_df['x'], session_df['y'])
        )
        .filter(
            (pl.col("shelter") == True) &
            (pl.col("barrier_present") == False)
        )
        # .with_columns(
        #     [
        #         ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        #         for col in ['x', 'y', 'dist']
        #     ]
        # )
    )

    return session_df











###########################################################################################
#                                    GET SPIKES DATA                                      #
###########################################################################################

def estimate_firing_rates_long_recursive(
    spikes: pl.LazyFrame,
    alpha: float,
    dt: float = 1.0,
    riemann: bool = True,
) -> pl.DataFrame:
    """
    Estimate firing rates from spike times using the causal kernel

        w(tau) = alpha^2 * tau * exp(-alpha * tau),   tau >= 0

    Parameters
    ----------
    spikes
        LazyFrame with columns:
        ['animal_idx', 'exp_idx', 'frame', 'unit_idx']
        Each row is one spike.
    alpha
        Kernel parameter (> 0).
    dt
        Duration of one frame in continuous time units.
    riemann
        If False, uses sampled kernel weights:
            w_k = alpha^2 * (k dt) * exp(-alpha k dt)
        If True, uses the Riemann-sum approximation to the integral:
            w_k = alpha^2 * (k dt) * exp(-alpha k dt) * dt

    Returns
    -------
    pl.DataFrame
        Long-format dataframe with columns:
        ['animal_idx', 'exp_idx', 'frame', 'unit_idx', 'rate']
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    required = {"animal_idx", "exp_idx", "frame", "unit_idx"}
    # schema may require a cheap resolution step on LazyFrame
    missing = required - set(spikes.collect_schema().names())
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One row per observed frame with spike count
    spike_counts_lf = (
        spikes
        .group_by(["animal_idx", "exp_idx", "unit_idx", "frame"])
        .len()
        .rename({"len": "spike_count"})
    )

    # Frame range for each experiment
    exp_bounds_lf = (
        spikes
        .group_by(["animal_idx", "exp_idx"])
        .agg(
            pl.col("frame").min().alias("frame_min"),
            pl.col("frame").max().alias("frame_max"),
        )
    )

    # Attach frame range to each unit-count row
    counts_with_bounds = (
        spike_counts_lf
        .join(exp_bounds_lf, on=["animal_idx", "exp_idx"], how="left")
        .sort(["animal_idx", "exp_idx", "unit_idx", "frame"])
        .collect()
    )

    # Also need units with zero spikes on some frames but existing in the experiment
    units = (
        spikes
        .select(["animal_idx", "exp_idx", "unit_idx"])
        .unique()
        .join(exp_bounds_lf, on=["animal_idx", "exp_idx"], how="left")
        .sort(["animal_idx", "exp_idx", "unit_idx"])
        .collect()
    )

    lam = np.exp(-alpha * dt)
    scale = (alpha ** 2) * (dt ** 2 if riemann else dt)

    # Build a lookup of sparse spike counts for each unit
    grouped_sparse = {
        (a, e, u): (frames.to_numpy(), counts.to_numpy())
        for (a, e, u), subdf in counts_with_bounds.group_by(
            ["animal_idx", "exp_idx", "unit_idx"], maintain_order=True
        )
        for frames, counts in [(subdf["frame"], subdf["spike_count"])]
    }

    out = []

    for row in units.iter_rows(named=True):
        a = row["animal_idx"]
        e = row["exp_idx"]
        u = row["unit_idx"]
        fmin = row["frame_min"]
        fmax = row["frame_max"]

        T = fmax - fmin + 1
        s = np.zeros(T, dtype=np.float64)

        key = (a, e, u)
        if key in grouped_sparse:
            frames, counts = grouped_sparse[key]
            s[frames - fmin] = counts

        # Recursive convolution
        u_state = 0.0
        q_state = 0.0
        rates = np.empty(T, dtype=np.float64)

        for t in range(T):
            prev_u = u_state
            u_state = lam * u_state + s[t]
            q_state = lam * q_state + lam * prev_u
            rates[t] = scale * q_state

        out.append(
            pl.DataFrame(
                {
                    "animal_idx": np.full(T, a),
                    "exp_idx": np.full(T, e),
                    "frame": np.arange(fmin, fmax + 1),
                    "unit_idx": np.full(T, u),
                    "rate": rates,
                }
            )
        )

    if not out:
        return pl.DataFrame(
            schema={
                "animal_idx": pl.Int64,
                "exp_idx": pl.Int64,
                "frame": pl.Int64,
                "unit_idx": pl.Int64,
                "rate": pl.Float64,
            }
        )

    return pl.concat(out, how="vertical")


def score_firing_rates(
    rates: pl.DataFrame
) -> pl.DataFrame:
    """
    Z-score firing rates 

    Parameters
    ----------
    rates
        DataFrame with columns:
        ['animal_idx', 'exp_idx', 'frame', 'unit_idx', 'rate']
        Each row is firing rate at a particular point in time.

    Returns
    -------
    pl.DataFrame
        Long-format dataframe with columns:
        ['animal_idx', 'exp_idx', 'frame', 'unit_idx', 'rate', 'zrate']
    """

    required = {"animal_idx", "exp_idx", "frame", "unit_idx", "rate"}
    # schema may require a cheap resolution step on LazyFrame
    missing = required - set(rates.collect_schema().names())
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # get mean, std of firing rate per unit
    stats = (
        rates
        .group_by('animal_idx', 'exp_idx', "unit_idx")
        .agg([
            pl.col("rate").mean().alias("mean_rate"),
            pl.col("rate").std().alias("std_rate"),
        ])
    )

    # compute zrate = (rate - mean_rate) / std_rate per unit (0 if std_rate==0)
    scores = (
        rates
        .join(stats, on=['animal_idx', 'exp_idx', "unit_idx"], how="left")   
        .with_columns(
            pl.when(pl.col("std_rate") == 0)
            .then(0)
            .otherwise((pl.col("rate") - pl.col("mean_rate")) / pl.col("std_rate"))
            .alias("zrate")
        )
        .select(['animal_idx', "exp_idx", "frame", "unit_idx", "zrate", "rate"])
    )

    return scores
    
    


def get_spikes_data(animal_idx, exp_idx, animal, exp_path, behave_df, alpha=10, dt=0.025):
    session_timesteps = (
        behave_df
        # .filter((pl.col("exp_idx") == exp_idx) & pl.col('animal_idx') == animal_idx)
        .select('frame')
        .sort('frame')
        .unique()
        .to_series()
    )

    spikes_lf = (
        pl.scan_csv(os.path.join(base_path, sub_path, animal, exp_path, "processed_data", "Processed_efizz_data"))
        .filter(
            pl.col("cluster_group").is_in(['good']),
            pl.col("spike_aligned_to_frame").is_in(session_timesteps.implode())
        )
        .select(['spike_aligned_to_frame', 'spike_clusters'])
        .rename({"spike_aligned_to_frame": "frame", "spike_clusters": "unit_idx"})
        .with_columns(
            (
                pl.lit(animal_idx).alias("animal_idx"),
                pl.lit(exp_idx).alias("exp_idx"),
                pl.col("frame").cast(pl.Int64)
            )
        )
    )

    rates_df = estimate_firing_rates_long_recursive(
        spikes=spikes_lf,
        alpha=alpha,
        dt=dt
    )

    scores_df = score_firing_rates(
        rates_df
    )

    scores_wide = (
        scores_df
        .drop('rate')
        .pivot(
            on="unit_idx",
            index=["animal_idx", "exp_idx", "frame"],
            values="zrate",
            aggregate_function="first",
        )
    )

    return scores_wide







###########################################################################################
#                                           MAIN                                          #
###########################################################################################

if __name__ == '__main__':

    animal_idx, exp_idx, animal, exp_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]

    metadata = {
        **dict(
            fps = 40,
            frame_time = 1 / fps,
            alpha = 1 / 0.1,
            min_dist = 20,
            length_scale = 5.0
        ),
        'notes': '''
            Using single units
            Filtered for barrier absent, shelter present, out of shelter, >20cm from shelter
            Firing rate computed with causal exponential kernel, z-scored
        ''',
        'error': None
    }

    path = Path(os.path.join(save_path, str(animal_idx), str(exp_idx)))
    path.mkdir(parents=True, exist_ok=True)

    def write_metadata():
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            f.write(serialize_metadata(metadata) + '\n')
            f.close()
        
    try:
        behave_df = get_behavioural_data(animal_idx, exp_idx, animal, exp_path, dt=metadata['frame_time'], min_dist=metadata['min_dist'], length_scale=metadata['length_scale'])
        spikes_df = get_spikes_data(animal_idx, exp_idx, animal, exp_path, behave_df, alpha=metadata['alpha'], dt=metadata['frame_time'])
    except Exception as e:
        print(e)
        metadata['error'] = str(e)
        write_metadata()
        quit()

    filtered_df = (
        behave_df
        .filter(
            pl.col("OutofshelterIdx") == True,
            pl.col("shelter") == True,
            pl.col("barrier_present") == False,
            pl.col("EscapePeriod") == False,
            pl.col("homingPeriod") == False,
            pl.col('far_from_shelter') == True
        )
        .select(['animal_idx', 'exp_idx', 'frame', 'hdir', 'hsa', 'x', 'y', 'dist', 'angular_speed', 'translational_speed'])
    )

    print('Behaviour:')
    print(behave_df.head())

    print('\nFiltered Behaviour:')
    print(filtered_df.head())

    print('\nSpikes:')
    print(spikes_df.head())

    save_df = (
        filtered_df
        .join(
            spikes_df,
            on=['animal_idx', 'exp_idx', 'frame'],
            how='inner'
        )
        .sort('frame')
    )

    write_csv(
        save_df, os.path.join(path, f'data.csv')
    )

    print('\nFinal:')
    print(save_df.head())

    write_metadata()

        

