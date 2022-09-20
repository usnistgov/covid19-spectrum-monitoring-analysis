#!/usr/bin/env python3

"""
Postprocessing calculations for swept power data.
"""

from numbers import Number
import os

os.environ["NUMEXPR_MAX_THREADS"] = "4"

import numpy as np
import pandas as pd
from pathlib import Path

import labbench as lb

# import sys
from collections import namedtuple

import dask
from dask import distributed
from dask import bag as db
import pickle
import time

import numexpr as ne

__all__ = [
    "to_single_parquet",
    "dask_sample_peak",
    "set_config",
    "sample_peak_along_time",
    "select_frequency",
    "dask_histogram",
    "hist_laxis",
    "config",
    "power_histogram_along_time",
    "dask_quiet_band",
    "by_frequency",
    "swept_power_dat_to_bags",
    "analyze_to_disk",
]

try:
    # imports when this is part of a library
    from . import munging, read_dat
except:
    # imports when this is executed as an example (see __main__ block below)
    import munging, read_dat

import warnings

warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="invalid value encountered")

config = {}

def dBtopow(x):
    # for large arrays, this is much faster than just writing the expression in python
    values = ne.evaluate('10**(x/10.)', local_dict=dict(x=x))
    
    if isinstance(x, pd.Series):
        return pd.Series(
            values,
            index=x.index
        )
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(
            values,
            index=x.index,
            columns=x.columns
        )
    else:
        return values

def powtodB(x, abs=True, eps=0):
    # for large arrays, this is much faster than just writing the expression in python
    if abs:
        values = ne.evaluate('10*log10(abs(x)+eps)', local_dict=dict(x=x, eps=eps))
    else:
        values = ne.evaluate('10*log10(x+eps)', local_dict=dict(x=x, eps=eps))

    if isinstance(x, pd.Series):
        return pd.Series(
            values,
            index=x.index
        )
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(
            values,
            index=x.index,
            columns=x.columns
        )
    else:
        return values


def set_config(config_module):
    """ `config_module` should be an imported module containing configuration constants
    """
    global config
    config = config_module


def _check_config():
    if config is None:
        raise ValueError(
            f"need to pass in an imported `config` module first with {repr(set_config)}(config)"
        )
                

def _concat_bags(bags):
    """ (delayed) concatenate bags into a dataframe
    """
    def bootstrapped_concat(df1, df2):
        """ concat that accepts None as an initial value
        """
        if df1 is None:
            return df2

        return pd.concat([df1,df2])

    return bags.fold(
        bootstrapped_concat, bootstrapped_concat, initial=None,
        split_every=16 # increase for memory efficiency, decrease for cpu efficiency?
    )  # concat dataframes by file, then by partition


def hist_laxis(df, n_bins, range_limits):
    """ compute a histogram along an axis. ref:
        https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
    """

    data = df
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    N = data.shape[-1]
    bins = np.linspace(R[0], R[1], n_bins + 1)
    data2D = data.reshape(-1, N)
    idx = np.searchsorted(bins, data2D, "right") - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins * np.arange(data2D.shape[0])[:, None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins * data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts


def power_histogram_along_time(
    pvt: pd.DataFrame,
    bounds: tuple((float, float)),
    resolution_db: float,
    resolution_sweeps: int,
    power_db_abs: bool,
    dtype="uint32",
):

    """ Wraps `hist_laxis into` a DataFrame output, parameterized on resolution and abscissa
        bounds.
        
        
        pvt: time series of power levels in linear units,
        bounds: tuple of (min, max) bounds for the histogram power level bins,
        resolution_db: step size of the histogram power level bins,
        resolution_sweeps: number of (time) indices to group into a single time bin,
        power_db_abs: whether to take an absolute value of linear power when converting to dB
        dtype: the integer data type to use for histogram counts
    
        :param pvt: `pd.DataFrame` instance, indexed as `pvt.index`, with values in dBm
    """

    # truncate to an integer number of sweep blocks
    pvt = powtodB(pvt, abs=power_db_abs)

    pvt = pvt.iloc[: resolution_sweeps * (len(pvt) // resolution_sweeps)]

    # use hist_laxis to compute the histogram at each time point
    shape = pvt.shape[0] // resolution_sweeps, pvt.shape[1] * resolution_sweeps
    reshaped = pvt.values.reshape(shape)
    n_bins = 1 + int((bounds[1] - bounds[0]) / resolution_db)
    h = hist_laxis(reshaped, n_bins, bounds).astype(dtype)

    # pack a DataFrame with the bin labels
    #     timestamps = pvt.index.get_level_values('Time')
    #     time_bins = pd.to_datetime(timestamps[::resolution_sweeps])
    power_bins = np.linspace(bounds[0], bounds[1], n_bins).astype("float64")
    df = pd.DataFrame(h, index=pvt.index[::resolution_sweeps], columns=power_bins)

    if "Sweep" in df.index.names:
        df.reset_index("Sweep", drop=True, inplace=True)

    return df


def sample_peak_along_time(
    pvt: pd.DataFrame, resolution_sweeps: int, power_db_abs: bool
):
    """ Divide the time series of power values `pvt` into bins, and then determine
        the largest in each bin.
        
        :resolution_sweeps: the number of frequency sweeps that forms the bin size in time
        :power_db_abs: whether to take an absolute value of linear power when converting to dB    
        
    
        :param pvt: `pd.DataFrame` instance, indexed as `pvt.index`, and values in dBm
    """

    # truncate to an integer number of sweep blocks
    pvt = pvt.iloc[: resolution_sweeps * (len(pvt) // resolution_sweeps)]

    # use hist_laxis to compute the histogram at each time point
    shape = pvt.shape[0] // resolution_sweeps, resolution_sweeps
    reshaped = pvt.values.reshape(shape)

    df = pd.DataFrame(
        reshaped.max(axis=1),
        index=pvt.index[::resolution_sweeps],
        columns=["Sample peak"],
    )

    if "Sweep" in df.index.names:
        df.reset_index("Sweep", drop=True, inplace=True)

    return powtodB(df, abs=power_db_abs)


def swept_power_dat_to_bags(dat_paths, subtract_ambient=None):
    """ Create dask bags representing the recorded band power.

        :returns: named tuple of (aperture average power, sample peaks per dwell)
    """
    _check_config()

    if subtract_ambient is None:
        subtract_ambient = config.DENOISE_METHOD

    if isinstance(dat_paths, tuple) and isinstance(dat_paths[0], db.Bag):
        # already packaged as db.Bag instances! pass them through
        return dat_paths


    DataPair = namedtuple("DataPair", ["aperture_average", "sample_peak"])


    def read(p, i):
        # add a bootstrapping delay to avoid triggering race conditions in google drive that deadlock the workstation
        try:
            bootstrap_count = config.DAT_FILES_PER_WORKER*config.CLUSTER_SETTINGS['n_workers']*config.CLUSTER_SETTINGS['threads_per_worker']
        except KeyError as exc:
            raise KeyError(f'{exc} - {config}')
        if i < bootstrap_count and i%config.DAT_FILES_PER_WORKER == 0:
            time.sleep(3*i/bootstrap_count)

        p = Path(p)
        try:
            tup = read_dat.swept_power(
                p,
                holdoff=config.SWEPT_POWER_HOLDOFF,
                excess_noise_subtraction_dB=config.POWER_EXCESS_NOISE_SUBTRACTION_DB,
                subtract_ambient=subtract_ambient,
                dof_factor=config.DOF_FACTOR,
            )
        except (IOError, ValueError) as ex:
            # log and skip when these exceptions are raised
            with open("ignored files.txt", "a") as fd:
                fd.write(f"{str(p)} read triggered {str(ex)}\n")
            return None

        aperture_average, metadata = tup

        times = aperture_average.index.get_level_values("Time").tz_localize(None)
        times = np.array(times, dtype="datetime64[us]")

        # these become columns
        reset_levels = 'Sweep', 'Frequency'

        aperture_average = read_dat._prepend_index_levels(
            aperture_average,
            Path=str(p)
        )
        aperture_average.reset_index('Time', drop=True, inplace=True)
        aperture_average.reset_index(reset_levels, inplace=True)
        aperture_average["Time"] = times

        sample_peak = pd.DataFrame(metadata["peak_power"])

        sample_peak = read_dat._prepend_index_levels(
            sample_peak,
            Path=str(p)
        )
        sample_peak.reset_index('Time', drop=True, inplace=True)
        sample_peak.reset_index(reset_levels, inplace=True)
        sample_peak["Time"] = times

        return DataPair(aperture_average=aperture_average, sample_peak=sample_peak)

    def select_dat_output(pair, field):
        return getattr(pair, field)

    npartitions = len(dat_paths)//config.DAT_FILES_PER_WORKER

    # a dask bag of the input data files
    dats = db.from_sequence([str(p) for p in dat_paths], npartitions=npartitions)
    indices = db.from_sequence(list(range(len(dat_paths))), npartitions=npartitions)

    # read each .swept_power.dat file, and filter out failures
    def validate_input(x):
        return x is not None

    pairs = dats.map(read, i=indices).filter(validate_input)

    # de-interleave the aperture_average and sample_peak data from each dat file
    aperture_average = pairs.map(select_dat_output, "aperture_average")
    sample_peak = pairs.map(select_dat_output, "sample_peak")

    return aperture_average, sample_peak


def occupancy_duration_by_threshold(dwells: pd.DataFrame, thresholds_dB: list, center_frequencies = 'all'):
    """ compute the occupancy time of transmission events within dwell windows in `dwells`.
    """
    # use the Path to determine the Site hash, and set the index we use for archival
    path = Path(dwells.index.get_level_values("Path")[0])
    dwells["Site"] = path.name.split(" ", 1)[0]

    dwells.reset_index(drop=True, inplace=True)
    dwells.set_index(["Frequency", "Site", "Time", "Sweep"], inplace=True)

    if center_frequencies != 'all':
        subset = set(center_frequencies).intersection(dwells.index.levels[0])
        dwells = dwells.loc[subset]

    time_elapsed = dwells.columns
    Ts=time_elapsed[1]-time_elapsed[0]

    nans = pd.DataFrame(
        np.full_like(dwells, np.nan),
        columns = time_elapsed + (time_elapsed[-1]-time_elapsed[0]+Ts),
        index = dwells.index
    )

    ret = {}

    for threshold_dB in thresholds_dB:
        wide = pd.concat([dwells, nans], axis=1)
        threshold = dBtopow(threshold_dB)

        i0, i1 = np.where((wide >= threshold)&(wide.shift(1,axis=1) < threshold))

        i0 = i0[:,np.newaxis] + np.zeros((1,dwells.shape[1]), dtype=int)
        i1 = i1[:,np.newaxis] + np.arange(dwells.shape[1], dtype=int)[np.newaxis,:]

        # sweep_timestamp + start_delay

        progression = pd.DataFrame(
            wide.values[i0,i1],

            # time elapsed since index timestamp
            columns = time_elapsed-time_elapsed[0],

            # offset indices by the start time in the dwell window
            index = wide.index[i0[:,0]].droplevel(['Sweep'])
        )
        progression.reset_index('Time', inplace=True)
        progression['Time'] = progression['Time'] + pd.TimedeltaIndex(wide.columns.values[i1[:,0]], unit='s')
        progression.set_index('Time', append=True, inplace=True)

        progression.mask(progression <= threshold, inplace=True)
        progression.mask(progression.cumsum(skipna=False, axis=1).isnull(), inplace=True)
        duration = progression.notnull().sum(axis=1)*Ts

        # select only occupancy events that stop before the dwell window
        complete_inds = i1[:,0]+duration/Ts < dwells.shape[1]-1
        duration = duration[complete_inds]

        ret[threshold_dB] = duration

    return pd.DataFrame(
        ret,
        columns = pd.Index(ret.keys(), name='Threshold (dB)')
    )


def by_frequency(power: pd.DataFrame, func: callable, *args, **kws):
    """ Evaluate ```func(power.loc[frequency], *args, **kws)``` for each entry in the 'Frequency'
        column of `power`. 
        
        :param power: a pd.DataFrame indexed on path (should all be the same), and columns
                      ['Frequency', 'Time', 'Sweep']

        :returns: a pd.DataFrame that consists of the concatenated results of `func`, with
                  a multilevel index with levels ['Frequency', 'Site', 'Time', 'Sweep']
    """

    # use the Path to determine the Site hash, and set the index we use for archival
    path = Path(power.index.get_level_values("Path")[0])
    power["Site"] = path.name.split(" ", 1)[0]

    try:
        power.reset_index(drop=True, inplace=True)
        power.set_index(["Frequency", "Site", "Time", "Sweep"], inplace=True)

    except AssertionError as ex:
        raise
        raise AssertionError(f"{ex.args[0]} (from input file at {path})")

    # This object configures the column and index parameters for calling `func`
    # for each unique Frequency. `group_keys=False` skips adding Frequency to the
    # index (it's already there.)
    gb = power.groupby("Frequency", group_keys=False)

    # Return a DataFrame comprising the return values from func calls
    # at each unique Frequency
    return gb.apply(func, *args, **kws)


def select_frequency(power: pd.DataFrame, fc: float):
    """ Evaluate ```func(power.loc[frequency], *args, **kws)``` for each entry in the 'Frequency'
        column of `power`. 
        
        :param power: a pd.DataFrame indexed on path (should all be the same), and columns
                      ['Frequency', 'Time', 'Sweep']

        :returns: a pd.DataFrame that consists of the concatenated results of `func`, with
                  a multilevel index with levels ['Frequency', 'Site', 'Time', 'Sweep']
    """

    # use the Path to determine the Site hash, and set the index we use for archival
    path = Path(power.index.get_level_values("Path")[0])
    power["Site"] = path.name.split(" ", 1)[0]

    try:
        power.reset_index(drop=True, inplace=True)
        power.set_index(["Frequency", "Site", "Time", "Sweep"], inplace=True)

    except AssertionError as ex:
        raise
        raise AssertionError(f"{ex.args[0]} (from input file at {path})")

    try:
        return power.loc[fc]
    except KeyError:
        return power.loc[[]]


def dask_histogram(inputs):
    """
        Prepare the computations for returning all aperture_average samples from the 2695 MHz
        quiet band. The return value can be used to compute the result as a pandas DataFrame as
        
        ```
            dask_histogram(inputs).compute()
        ```
        
        or computed in parallel with other delayed operations as
        
        ```
            other1 = <other dask delayed instance>
            other2 = <another dask delayed instance>
            
            dask.compute([
                dask_histogram(inputs),
                other1,
                other2
            ])
        ```
        
         :inputs: either a list of paths to *.swept_power.dat files, or the dask DataFrame instances
                         that result from loading them        
        
        :returns: `dask.delayed` instance
    """
    _check_config()

    averages, sample_peaks = swept_power_dat_to_bags(inputs)

    # compute and save histograms
    # hist_cols = list(range(config.HISTOGRAM_POWER_LOW, config.HISTOGRAM_POWER_HIGH + 1))

    lb.logger.warning(f'POWER_DB_ABS = {config.POWER_DB_ABS}')

    bags = averages.map(
        by_frequency,
        power_histogram_along_time,
        bounds=(config.HISTOGRAM_POWER_LOW, config.HISTOGRAM_POWER_HIGH),
        resolution_db=config.HISTOGRAM_RESOLUTION_DB,
        resolution_sweeps=config.HISTOGRAM_RESOLUTION_SWEEPS,
        power_db_abs=config.POWER_DB_ABS,
        dtype="uint32",
        # tell dask the return columns that it should expect
        #         meta=[(c,'uint32') for c in hist_cols],
    )

    return _concat_bags(bags)

def dask_durations(inputs):
    """
        Prepare the computations for computing dwell occupancies. The return value can be used to compute the result as a pandas DataFrame as
        
        ```
            dask_histogram(inputs).compute()
        ```
        
        or computed in parallel with other delayed operations as
        
        ```
            other1 = <other dask delayed instance>
            other2 = <another dask delayed instance>
            
            dask.compute([
                dask_durations(inputs),
                other1,
                other2
            ])
        ```
        
         :inputs: either a list of paths to *.swept_power.dat files, or the dask DataFrame instances
                         that result from loading them        
        
        :returns: `dask.delayed` instance
    """
    _check_config()

    averages, sample_peaks = swept_power_dat_to_bags(inputs)

    # compute and save histograms
    # hist_cols = list(range(config.HISTOGRAM_POWER_LOW, config.HISTOGRAM_POWER_HIGH + 1))

    bags = averages.map(
        occupancy_duration_by_threshold,
        thresholds_dB = config.OCCUPANCY_THRESHOLDS_DB,
        center_frequencies = config.OCCUPANCY_CENTER_FREQUENCIES
    )

    return _concat_bags(bags)

def dask_quiet_band(inputs):
    """ 
        Prepare the computations for returning all aperture_average samples from the 2695 MHz
        quiet band. The return value can be used to compute the result as a pandas DataFrame as
        
        ```
            dask_sample_peak(inputs).compute()
        ```

        or computed in parallel with other delayed operations as
        
        ```
            other1 = <other dask delayed instance>
            other2 = <another dask delayed instance>
            
            dask.compute([
                dask_quiet_band(inputs),
                other1,
                other2
            ])
        ```
        
        :inputs: either a list of paths to *.swept_power.dat files, or the dask DataFrame instances
                         that result from loading them        
        
        :returns: `dask.delayed` instance
    """

    _check_config()

    averages, sample_peaks = swept_power_dat_to_bags(
        inputs, subtract_ambient=False
    )

    bags = averages.map(select_frequency, 2695.0,)

    return _concat_bags(bags)

def dask_export_dataset(inputs, folder, site_number_map):
    # this mkdir() doesn't seem to be fully multiprocessing safe, so give it a few attempts
    @lb.retry((OSError,FileExistsError), tries=8, delay=1)
    def export_to(df, export_path):
        export_path.parent.mkdir(exist_ok=True, parents=True)

        (
            df
            .set_index(['Time','Frequency'], drop=True) # clears out (sweep number, file)
            .to_csv(
                export_path,
                chunksize=df.shape[0],
                float_format='%.3g',
                compression=dict(method='gzip', compresslevel=5)
            )
        )

    def write_csv(df):
        dat_path = Path(df.index.get_level_values('Path')[0])
        site_name = site_number_map[dat_path.parent.name]

        filename = (dat_path.name.split('.',1)[0]+'.csv.gz').replace(dat_path.parent.name, site_name)
        export_path = Path(folder)/site_name/filename

        export_to(df, export_path)

    averages, sample_peaks = swept_power_dat_to_bags(inputs)

    return averages.map(write_csv)

def dask_sample_peak(inputs):
    """ :inputs: either a list of paths to *.swept_power.dat files, or the dask DataFrame instances
                 that result from loading them
        
        Prepare the computations for rolling max of sample peaks. The return value can be used to
        compute the result as a pandas DataFrame as
        
        ```
            dask_sample_peak(inputs).compute()
        ```
        
        or computed in parallel with other delayed operations as
        
        ```
            other1 = <other dask delayed instance>
            other2 = <another dask delayed instance>
            
            dask.compute([
                dask_sample_peak(inputs),
                other1,
                other2
            ])
        ```
        
        :returns: `dask.delayed` instance
    """

    _check_config()

    averages, sample_peaks = swept_power_dat_to_bags(inputs)

    bags = sample_peaks.map(
        by_frequency,
        sample_peak_along_time,
        resolution_sweeps=config.HISTOGRAM_RESOLUTION_SWEEPS,
        power_db_abs=config.POWER_DB_ABS,
    )
    
    return _concat_bags(bags)


@dask.delayed
def to_single_parquet(df, path):
    """ delayed write of the pandas DataFrame as a parquet file.
    """

    # code the user hashes to a site ID number
    i_site = df.index.names.index('Site')

    levels = list(df.index.levels)
    output_names = list(range(len(levels[i_site])))

    for i in range(len(levels[i_site])):
        if 'hospital' in levels[i_site][i]:
            output_names[i] = 'hospital'
        else:
            output_names[i] = f'{output_names[i]:02g}'

    pd.Series(
        levels[i_site],
        index=output_names
    ).to_csv(
        path.parent/'site_hash_mapping.csv'
    )

    levels[i_site] = pd.Index(output_names)

    index = pd.MultiIndex(
        levels=levels,
        codes=df.index.codes,
        names=df.index.names
    )

    df = pd.DataFrame(
        df.values,
        # to_parquet supports only string column names
        columns=df.columns.astype("str"),
        # but it does support MultiIndex and timestamp datatypes!
        index=index,
    )
    
    df.to_parquet(
        path,
        allow_truncated_timestamps=True,
        row_group_size=100000 # pd.read_parquet() balloons in memory unless we set this
    )


def analyze_to_disk(input_dat_files, data_root, histogram=None, quiet_band=None, sample_peak=None, export_denoised=None, export_samples=None, durations=None):
    """
        :returns: None
    """

    # generate the dask DataFrame instances up front. Otherwise,
    # dask doesn't recognize that the analyses should share
    #  the same file read (and groupby) operations, and wastefully
    # repeats them
    if histogram is not None or sample_peak is not None or export_denoised is not None or durations is not None:
        input_bags = swept_power_dat_to_bags(input_dat_files)

    if quiet_band is not None or export_samples is not None:
        input_bags_raw = swept_power_dat_to_bags(input_dat_files, subtract_ambient=False)

    # generate the sequence of file save operations, which are chained
    # to analysis operations. these are returned as a list of dask.delayed instances
    delayed = []
    if histogram is not None:
        delayed += [to_single_parquet(dask_histogram(input_bags), histogram)]

    if export_samples:
        mapping_dict = pd.read_csv(Path(data_root)/'site_hash_mapping.csv').set_index('Site').iloc[:,0].to_dict()
        delayed += [dask_export_dataset(input_bags_raw, export_samples, mapping_dict)]

    if export_denoised:
        mapping_dict = pd.read_csv(Path(data_root)/'site_hash_mapping.csv').set_index('Site').iloc[:,0].to_dict()
        delayed += [dask_export_dataset(input_bags, export_denoised, mapping_dict)]

    if quiet_band is not None:
        # need to repeat for quiet_band in order to disable background noise subtraction
        delayed += [to_single_parquet(dask_quiet_band(input_bags_raw), quiet_band)]

    if sample_peak is not None:
        delayed += [to_single_parquet(dask_sample_peak(input_bags), sample_peak)]

    if durations is not None:
        delayed += [to_single_parquet(dask_durations(input_bags), durations)]

    if len(delayed) == 0:
        raise ValueError("pass at least 1 file name to perform analysis")
    else:
        if export_denoised or export_samples:
            # workers sometimes take so long to dump its csv file that timeout exceptions
            # are raised. increasing this here addresses the problem, but unfortunately
            # could lead to very long delay before other useful exceptions are raised
            params = dict(timeout=900)
        else:
            params = dict()

        dask.compute(delayed, **params)


if __name__ == "__main__":
    # TODO: add an example here
    pass
