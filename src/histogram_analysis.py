import numpy as np
import pandas as pd
import numexpr as ne
from environment import time_format
from dwell_analysis import dBtopow, powtodB

SAMPLES_PER_DWELL = 210 #  TODO: don't hardcode timing parameters
SAMPLES_PER_HISTOGRAM = 60*SAMPLES_PER_DWELL

import numexpr as ne

def rebin_time_scale(h, scale_period, remove_date=True):
    idx = h.index.get_level_values('Time').floor('10T')
    if remove_date:
        idx = idx-pd.DatetimeIndex(idx.date)
    h_on_period = h.groupby(['Site', idx]).sum()
    h_on_period.index.names = ['Site', 'Time']
    return h_on_period

def loc_by_fc_spec(hists, fc_or_list):
    """
    equivalent to ```hists.loc[fc_or_list]```, except that each fc
    is mapped from the test specification to the actual frequency reported by the sensor.
    """
    fc_map = dict(zip(np.round(hists.index.levels[0],1),hists.index.levels[0]))
    if hasattr(fc_or_list, '__iter__'):
        return hists.loc[[fc_map[fc] for fc in fc_or_list]]
    else:
        return hists.loc[fc_map[fc_or_list]]

def date_bounds(hists):
    """ return a tuple of strings (start_date, end_date)
    """
    start_date = hists.index.levels[hists.index.names.index('Time')].min()
    start_date = str(start_date).split(' ', 1)[0]

    end_date = hists.index.levels[hists.index.names.index('Time')].max()
    end_date = str(end_date).split(' ', 1)[0]

    return f'from {start_date} through {end_date}'

def rcumsum(x, copy=True, power_fix=True):
    """ cumsum from the end to the beginning of axis 1
    """
    df = None

    if hasattr(x, 'values'):
        is_df = True
        df, x = x, x.values

    if copy:
        x = x.copy()

    if power_fix:
        x[:,0] += SAMPLES_PER_HISTOGRAM - x.sum(axis=1)
    
    x = np.cumsum(x[:,::-1], axis=1)[:,::-1]

    # fill in values that were less than 0 dB (i.e., "negative power") as below all values of x
    # x = x + (SAMPLES_PER_HISTOGRAM-x[:,0,np.newaxis])

    if df is not None:
        return pd.DataFrame(
            x,
            index=df.index,
            columns=df.columns
        )
    else:
        return x

def contiguous_segments(df, index_level, threshold=7, relative=True):
    """Split `df` into a list of DataFrames for which the index values
    labeled by level `index_level`, have no discontinuities greater
    than threshold*(median step between index values).
    """
    delta = pd.Series(df.index.get_level_values(index_level)).diff()
    if relative:
        threshold = threshold * delta.quantile(0.1)
    i_gaps = delta[delta > threshold].index.values
    i_segments = [[0] + list(i_gaps), list(i_gaps) + [None]]

    return [df.iloc[i0:i1] for i0, i1 in zip(*i_segments)]

def rcumsum_on_dwell_norm(hists, median_period, dwell_stat='min'):
    """ compute and return the (not-normalized) survival function
    of the power normalized in each dwell window.
    """
    dwell_mode = compute_dwell_mode(
        hists,
        rolling_stat_period=median_period,
        rolling_stat_name=dwell_stat,
        by=['Frequency','Site']
    )

    h_normed = normalize_power_per_dwell(hists, dwell_mode)

    x = rcumsum(h_normed.values)

    h_normed_rcum = pd.DataFrame(
        x,
        index = h_normed.index,
        columns = h_normed.columns
    )

    return h_normed_rcum, dwell_mode

def read_histogram(path, site=None, year=None, month=None, day=None, duration:dict=None, end=None):
    if year:
        start = pd.Timestamp(year=year, month=month or 1, day=day or 1)
    
    filters = []
    
    if site is not None:
        filters.append(['Site','=',site])

    if year is not None:
        start = pd.Timestamp(year=year, month=month or 1, day=day or 1)        
        filters.append(['Time', '>=', start])
        
    if duration is not None:
        if year is None:
            raise ValueError(f"duration argument not supported unless a start date is passed")
        filters.append(['Time', '<', start+pd.DateOffset(**dict(duration))])
        
    elif end is not None:
        filters.append(['Time', '<', pd.Timestamp(end)])

    hists = pd.read_parquet(
        path,
        filters=filters or None,
        use_threads=False
    )

    if site is not None:
        hists.reset_index('Site', drop=True, inplace=True)

    hists.columns = hists.columns.astype('float32')
    
    return hists

def mask_survival_near_noise(survival, noise_survival, margin_dB=6):
    """ when called by groupby.apply, mask survival function when less than
        the shielded histogram median (in dB) + margin_dB

        Arguments:

            noise_hists: a histogram dataframe (with first level 'Frequency')
                containing histograms of noise

    """
    fc = survival.index[0][survival.index.names.index('Frequency')]
    threshold = np.abs(noise_survival.loc[fc]-0.5).idxmin() + margin_dB

    return pd.DataFrame(
        [survival.columns < threshold],
        columns=survival.columns
    )
    survival = survival.astype('float').iloc[0]
    survival.values[:,survival.columns < threshold] = np.nan
    return survival.isnull()

def normalize_by_first_row(df):
    return pd.DataFrame(
        df.values/df.values[0,:][np.newaxis,:],
        index=df.index,
        columns=df.columns
    ) 

def rcum_quantile(df_rcum, q, axis=0):
    return np.abs(df_rcum-q).idxmin(axis=axis)

def rolling_stat(df, period, stat_name='min'):
    non_time_levels = [n for n in df.index.names if n != 'Time']
    
    rolled = (
        df
        .reset_index(non_time_levels, drop=True)
        .sort_index()
        .rolling(period, min_periods=1)
        .agg(stat_name)
        .values
    )

    # to use a rolling time period (e.g., '1H') with histogram dataframes,
    # we need to clear frequency out of the index
    return pd.Series(rolled, index=df.index)

def compute_dwell_mean(hists, accept_quantile=None, rolling_stat_period: float = None, rolling_stat_name='min', by='Frequency'):
    power = dBtopow(hists.columns.values)
    counts = hists.values

    if accept_quantile is not None:
        # drop quantiles
        counts = counts.copy()
        h_cum = hists.cumsum(axis=1).astype('float').values
        h_cum[:] /= h_cum[:,-1,None]
        th_low = np.abs(h_cum-(0.5-accept_quantile/2)).argmin(axis=1)[:,np.newaxis]
        th_high = np.abs(h_cum-(0.5+accept_quantile/2)).argmin(axis=1)[:,np.newaxis]

        ii = np.arange(h_cum.shape[1])

        counts[(ii<th_low)|(ii>=th_high)] = 0
    
    series_list = []

    dwell_mean = np.sum(power[np.newaxis,:]*counts,axis=1)/np.sum(counts,axis=1)


    dwell_mean = pd.Series(
        powtodB(dwell_mean),
        index = hists.index,
        name='Mean'
    )
    
    if rolling_stat_period is not None:
        if by is not None:
            dwell_mean = (
                dwell_mean
                .groupby(by, sort=False)
                .apply(rolling_stat, rolling_stat_period, stat_name=rolling_stat_name)
                .bfill()
            )
        else:
            dwell_mean = rolling_stat(dwell_mean, rolling_stat_period)

    return dwell_mean
        
def compute_dwell_mode(hists, rolling_stat_period: float = None, rolling_stat_name='min', by='Frequency'):
    counts = hists.values

    dwell_mode = pd.Series(
        hists.columns[np.argmax(counts, axis=1)],
        index=hists.index,
        name='Mode'
    )
    
    if rolling_stat_period is not None:
        if by is not None:
            dwell_mode = (
                dwell_mode
                .groupby(by, sort=False)
                .apply(rolling_stat, rolling_stat_period, stat_name=rolling_stat_name)
                .bfill()
            )
        else:
            dwell_mode = rolling_stat(dwell_mode, rolling_stat_period)

    return dwell_mode

def compute_dwell_median(hists, rolling_stat_period: float = None, rolling_stat_name='min', by=None):
    counts = hists.values

    power_step = pd.Series(hists.columns).diff().median()
    cs = hists.values.cumsum(axis=1)
    cs = cs/cs[:,-1][:,np.newaxis]

    all_i = np.arange(cs.shape[0])
    idx = np.abs(cs-0.5).argmin(axis=1)

    dwell_median = pd.Series(
        hists.columns[idx].values+power_step,
        index=hists.index,
        name=f'Median'
    )

    if rolling_stat_period is not None:
        if by is not None:
            dwell_median = (
                dwell_median
                .groupby(by, sort=False)
                .apply(rolling_stat, rolling_stat_period, stat_name=rolling_stat_name)
                .bfill()
            )
        else:
            dwell_median = rolling_stat(dwell_median, rolling_stat_period)

    return dwell_median


def normalize_power_per_dwell(hists: pd.DataFrame, power_norm: pd.Series):
    """ Compute a new pd.DataFrame in which the power axis (columns)
    has been normalized by the supplied pd.Series at each dwell window.
    """
    power_axis = hists.columns.values
    counts = hists.values
    power_step = pd.Series(power_axis).diff().median()    
    
    idx_norms = np.abs(power_axis[np.newaxis,:]-power_norm.values[:,np.newaxis]).argmin(axis=1)
    new_bins = power_axis[np.newaxis,:]-power_axis[idx_norms][:,np.newaxis]
    new_columns = np.arange(new_bins.min().min(), new_bins.max().max()+power_step, power_step)
    new_bin_inds = ((new_bins-new_columns[0])/power_step).astype('int64')
    
    hists_relative = np.zeros((hists.shape[0], len(new_columns)))
    np.put_along_axis(hists_relative, new_bin_inds, counts, axis=1)

    hists_relative = pd.DataFrame(
        hists_relative,
        columns=new_columns,
        index=hists.index
    )
    
    return hists_relative

def hists_summary_table(hists, site=None):
    center_frequencies = hists.index.levels[0].values
    
    segments = contiguous_segments(hists, "Time", threshold=1.5)    

    average_time_step = (
        segments[0]
        .reset_index('Time')['Time']
        .groupby('Frequency')
        .diff()
        .mean()
    )

    df = pd.DataFrame(
        [{
            'Average time step': pd.Timedelta(average_time_step, unit='s').round('s'),
            'Frequency points': len(center_frequencies),
            'Number of histograms': hists.shape[0],
            'Start': pd.Series(hists.index.get_level_values('Time')).min().strftime(time_format),
            'End':pd.Series(hists.index.get_level_values('Time')).max().strftime(time_format),
        }],
        index=['']
    )

    if site is not None:
        df['Site'] = site

    return df.T

def _label_segments(df, index_level, threshold=7, relative=True):
    time_delta = pd.Series(df.index.get_level_values(index_level)).diff()
    
    if relative:
        threshold = threshold * time_delta.median()    

    df = df.copy()
    df['Segment'] = (
        (time_delta>threshold)
        .astype('int')
        .cumsum()
        .values # assignment to the Series fails because indices don't match
    )

    df.set_index('Segment', append=True, inplace=True)
    return df

def segment_timestamps(df, index_level='Time', threshold=7, relative=True):
    """ return a DataFrame containing a view on the values in df, and
    index with an additional 'Segment' level that gives an index for groups of contiguous
    gap-free readings.
    """
    segmented_index = (
        df
        .index.to_frame()
        .drop(df.index.names,axis=1)
        .groupby(['Site','Frequency'], as_index=False)
        .apply(
            _label_segments,
            index_level=index_level,
            threshold=threshold,
            relative=relative
        )
        .index
    )
    
    return pd.DataFrame(
        df.values,
        index=segmented_index,
        columns=df.columns
    )
