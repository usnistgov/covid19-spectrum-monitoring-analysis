import click
import sys

if "src" not in sys.path:
    sys.path.insert(1, "src")

import config, munging, read_dat, dwell_analysis

from dask import distributed
import dask
import labbench as lb
from pathlib import Path
import os
import time


@click.command(
    help="Process an input directory that contains a swept-power spectrum monitoring "
    "dataset, and write the analysis results into parquet files in DATA_ROOT."
    "Detailed settings may be adjusted in config.py."
)
@click.argument(
    "data-root",
    default=config.DEFAULT_DATA_ROOT,
    required=False,
    type=click.Path(exists=True),
)
@click.option("--quiet-band", is_flag=True, help='output "quiet band.parquet"')
@click.option("--histogram", is_flag=True, help='output "histogram.parquet"')
@click.option("--peaks", is_flag=True, help='output "peaks.parquet"')
@click.option("--time-series", is_flag=True, help='output "time series" directory')
@click.option("--time-series-denoised", is_flag=True, help='output "time series denoised"')
@click.option("--durations", is_flag=True, help='output "occupancy durations.parquet"')
def run(data_root, quiet_band, histogram, peaks, time_series, time_series_denoised, durations):
    data_root = Path(data_root)
    EXPORT_DATA_ROOT = Path(config.EXPORT_DATA_ROOT)

    # apply the settings from the config file
    dwell_analysis.set_config(config)

    lb.logger.warning('access client')

    # get a dask cluster client, attempting to start one locally if needed
    try:
        client = distributed.Client(config.CLUSTER_SETTINGS["host"], timeout="5s")
        lb.logger.warning('success')
    except OSError:
        lb.logger.warning('start client')
        # guess there isn't one already running, so start a cluster
        cluster = distributed.LocalCluster(**config.CLUSTER_SETTINGS)
        client = distributed.Client(config.CLUSTER_SETTINGS["host"])

    print(f"dashboard address: {str(cluster.dashboard_link)}")

    with lb.stopwatch("find .swept_power.dat files"), dask.config.set(
        scheduler="single-threaded"
    ):
        dat_paths = munging.find_swept_power_monitoring(
            data_root, min_sweeps=config.SWEPT_POWER_MIN_SWEEPS
        )

    print(f"i found {len(dat_paths)} good paths")

    with lb.stopwatch(f"analyze {len(dat_paths)} files"), client:
        t0 = time.perf_counter()

        dwell_analysis.analyze_to_disk(
            dat_paths,
            data_root=data_root,
            histogram = data_root / "histogram.parquet" if histogram else None,
            quiet_band = data_root / "quiet band.parquet" if quiet_band else None,
            sample_peak = data_root / "sample peak.parquet" if peaks else None,
            durations = data_root / "occupancy durations.parquet" if durations else None,
            export_denoised = EXPORT_DATA_ROOT / "time series denoised" if time_series_denoised else None,
            export_samples = EXPORT_DATA_ROOT / "time series" if time_series else None,
        )

    # performance info
    print(
        f"* {len(cluster.worker_spec)} procs "
        f"{cluster.worker_spec[0]['options']['nthreads']} threads "
        f"{len(dat_paths)} files "
        f'{os.environ["NUMEXPR_MAX_THREADS"]} numexpr threads '
        f"{len(dat_paths)/(time.perf_counter()-t0):0.2f} files/s"
    )


if __name__ == "__main__":
    run()
