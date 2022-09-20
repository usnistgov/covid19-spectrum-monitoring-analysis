from os import cpu_count
from socket import gethostname

# Default path for command line tools
DEFAULT_DATA_ROOT = r"G:\Shared drives\COVID-19 Spectrum Monitoring\Data"
# DEFAULT_DATA_ROOT = r'D:\dkuester\covidscratch'

EXPORT_DATA_ROOT = r"G:\Shared drives\COVID-19 Spectrum Monitoring Data Export"
# EXPORT_DATA_ROOT = r"D:\dkuester\covidscratch-export"


# If False: then calibrated noise power in dB is 10*log10(pow),
#           resulting in -inf for readings weaker than the calibrated levels.
#    True: calibrated noise power in dB is 10*log10(abs(pow)), which
#           distorts the eventual distribution
POWER_DB_ABS = False

# This scaling will be applied to the calibrated ambient noise level before subtraction
# when applying calibration to the raw data files. Increasing to slightly above
#
POWER_EXCESS_NOISE_SUBTRACTION_DB = (
    0  # <- experimented with this value for power subtraction
)

# How to attempt denoising to improve SNR.
# This should be 'power' 'magnitude', or 'distribution'; or False to disable subtraction
DENOISE_METHOD = "power"

# Adjustment factor for the degrees of freedom in the assumed
# noise distribution
DOF_FACTOR = 1

# low and high bounds for the power levels that make up the columns in 'histogram.parquet'
HISTOGRAM_POWER_LOW = -140  # dBm
HISTOGRAM_POWER_HIGH = -10  # dBm

# the power level increment for the column index computed for 'histogram.parquet'
HISTOGRAM_RESOLUTION_DB = 0.25

# each time bin will aggregate this many frequency sweeps in 'histogram.parquet'
HISTOGRAM_RESOLUTION_SWEEPS = 60  # 60 == around 10 minutes

# Initial swept power delay before accepting samples from the swept power data files
# in sec., between 0 and 0.3 (the dwell window duration
SWEPT_POWER_HOLDOFF = 0.195  # tested values include .05 and 0.195 - 0.195 avoids leftover "junk" from prior dwells

# if swept_power_max_files is None:
#     all swept power files are analyzed
# if swept_power_max_files > 0:
#     no more than this many swept power files will be analyzed (for debug)
SWEPT_POWER_MAX_FILES = None

# *.swept_power.dat files need at least this many frequency sweeps,
# otherwise they are ignored
SWEPT_POWER_MIN_SWEEPS = 2 * HISTOGRAM_RESOLUTION_SWEEPS

# The power level thresholds to use evaluating occupancy
OCCUPANCY_THRESHOLDS_DB = -75, -70, -65, -60, -55, -50

# Allowlist of center frequencies at which to evaluate occupancy
OCCUPANCY_CENTER_FREQUENCIES = [
    701.5,
    709.0,
    782.0,
    821.2999877929688,
    842.5,
    2412.0,
    2437.0,
    2462.0,
    2695.0,
    5170.0,
    5190.0,
    5210.0,
    5230.0,
    5240.0,
    5775.0,
    5795.0
]

# local computation and data parameters
import os

CLUSTER_SETTINGS = dict(
    # limiting this can help to avoid system hangs caused by google drivefs
    n_workers=10,
    threads_per_worker=2,
    host=f"tcp://{gethostname()}:8786",
    dashboard_address=f"{gethostname()}:8787",
    processes=True,  # set False to enable single-process for easier debugging
    local_directory=r"C:\users\dkuester\temp",
    memory_limit='210GB', # a few tasks need workers to accommodate writing large blocks of data to disk
)

# Each DAT file occupies around ~330MB on disk and in memory
# Lower bound:
#    when this = 1, there is a lot of overhead spinning up new processes, and google drive sometimes deadlocks
#    due to many processes hitting the cloud drive simultaneously.
# Upper bound:
#    Watch total memory consumption, roughly 330 MB*n_workers*threads_per_worker
#    Large values make it difficult to gauge progress in the dask dashboard
#
# A value of 5 has worked safely so far on a desktop and a high-end workstation
DAT_FILES_PER_WORKER = 1

# file type for reports. 'pdf' is tempting but ironically doesn't embed properly
# in pdf format
FIGURE_FORMAT = 'svg'

# these report notebooks are run when reports.py is called without arguments.
# the paths are taken relative to EXPORT_DATA_ROOT/'Analysis'
ALL_REPORTS = (
    # 'reports/window_2020-06--2020-07.ipynb',
    # 'reports/window_2020.ipynb',
    # 'reports/overview.ipynb',
    # 'reports/by_site.ipynb',
    # 'reports/occupancy_durations.ipynb',
    # 'reports/by_site_and_month.ipynb'
)

# a .tex file is generated for each glob in this sequence.
# the paths are taken relative to EXPORT_DATA_ROOT/'Analysis'
EXPORT_FIGURES = [
    # 'figures/overview*',
    'figures/occupancy_duration*',
    # 'figures/sanjole*',
    # 'by_site/figures/site_hospital*',
    # 'by_site/figures/site_08*',
] #+ [f'by_site/figures/site_{i:02g}*' for i in range(8,13)]