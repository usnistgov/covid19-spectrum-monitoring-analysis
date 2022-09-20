from environment import *
from matplotlib import ticker
import histogram_analysis
from textwrap import wrap

from histogram_analysis import SAMPLES_PER_HISTOGRAM


import datetime

power_label = r'Band power (dBm/4 MHz)'

def subplot_title(ax, text, **kws):
    """ draw an axis title centered on the full subplot rather than the plot area
    """
    def get_bbox_coord(obj, attr):
        try:
            return getattr(obj.get_window_extent(), attr)
        except:
            return None

    children = list(ax.get_children()) + [ax.yaxis.get_label()]

    x0 = min([get_bbox_coord(child, 'x0') for child in children if get_bbox_coord(child, 'x0')])
    x1 = max([get_bbox_coord(child, 'x1') for child in children if get_bbox_coord(child, 'x1')])
    y = ax.get_window_extent().y0
    x,_ = ax.transAxes.inverted().transform([0.5*(x0+x1), y])
    ax.set_title(text, x=x, **kws)


class ConciseDateFormatter(matplotlib.dates.ConciseDateFormatter):
    def format_ticks(self, values):
        labels = super().format_ticks(values)

        if self.offset_string:
            labels[-1] = labels[-1] + '\n' + self.offset_string
            self.offset_string = ''

        return labels

class ForceHoursFormatter(matplotlib.ticker.ScalarFormatter):
    def format_ticks(self, values):
        labels = super().format_ticks(values)
        return [str(l) + ':00' for l in labels]


def xaxis_concise_dates(fig, ax, adjacent_offset: bool=False):
    """ fuss with the dates on an x-axis.
    """
    formatter = ConciseDateFormatter(
        matplotlib.dates.AutoDateLocator(),
        show_offset=True)

    formatter.offset_formats = ['',
                                '%Y',
                                '%Y',
                                '%b %Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                # '%d %b %Y %H:%M'
                                ]
    # if adjacent_offset:
    # xticks(rotation=0, ha='right')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)
    # offset_text = ax.xaxis.get_offset_text().get_text()
    # print('offset text is ', offset_text)
    # last_xlabel = ax.get_xticklabels()[-1]
    # print('last xlabel is ', last_xlabel)
    # last_xlabel.set_text(last_xlabel.get_text()+'\n'+offset_text)
    # print('last xlabel is ', last_xlabel)

    # if adjacent_offset:
    #     labels = [item.get_text() for item in ax.get_xticklabels()]
    #     labels[0] = f'{formatter.get_offset()} {labels[0]}'
    #     ax.set_xticklabels(labels)

    #     dx = 5/72.; dy = 0/72. 
    #     offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    #     for label in ax.get_xticklabels():
    #         label.set_transform(label.get_transform() + offset)

    return ax

def set_axis_caption(ax, text, subtext=None):
    label = ax.xaxis.get_label()
    text = '\n'.join(['\n'.join(wrap(t, 50)) for t in text.split('\n')])

    label_text = label.get_text()
    if len(label_text)>0:
        label_text = label_text + '\n————————————————\n'

    label.set_text(label_text + text)


def plot_power_histogram_heatmap(
    hist_by_time: pd.DataFrame,
    title: str,
    bounds: tuple,  # (low, high)
    warn: bool = False,
    contiguous_threshold=2,
    log_counts=True,
    ax=None,
    cbar=True,
    vmax=None,
    clabel='#/bin'
):

    """plot power histograms from the spectrum along the time axis."""

    hist_by_time = hist_by_time.loc[:, float(bounds[0]) : float(bounds[1])]

    if ax is None:
        fig, ax = subplots()
    else:
        try:
            fig = ax.get_figure()
        except:
            raise ValueError(str(locals()))

    if hist_by_time.shape[0] == 0:
        raise EOFError

    index_type = type(hist_by_time.index[0])

    # elif issubclass(index_type, pd.Timedelta):
    #     pass
    # else:
    #     raise ValueError(
    #         f"don't know how to handle index type {index_type} for 2D histogram over time"
    #     )

    # quantize the color map levels to the number of bins
    cmap = matplotlib.cm.get_cmap("magma")
    if hist_by_time.shape[1] < cmap.N:
        subset = np.linspace(0, len(cmap.colors) - 1, hist_by_time.shape[1], dtype=int)
        newcolors = np.array(cmap.colors)[subset].tolist()
        cmap = matplotlib.colors.ListedColormap(newcolors)
        cmap.set_bad("0.95")

    if log_counts:
        if hist_by_time.values.dtype == dtype('int64'):
            plot_norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax or hist_by_time.max().max())
        else:
            plot_norm = matplotlib.colors.LogNorm(vmin=hist_by_time[hist_by_time>0].min().min(), vmax=vmax or hist_by_time.max().max())
    else:
        plot_norm = None

    if issubclass(index_type, pd.Timestamp):
        # break into contiguous segments so that matplotlib will not project lines across
        # missing data
        segments = histogram_analysis.contiguous_segments(
            hist_by_time, "Time", threshold=contiguous_threshold
        )

        for hist_sub in segments:
            c = ax.pcolormesh(
                hist_sub.index.values,
                hist_sub.columns.values,
                hist_sub.T.values,
                cmap=cmap,
                norm=plot_norm,
                rasterized=True,
                vmax=vmax
            )

    elif issubclass(index_type, pd.Timedelta):
        c = ax.pcolormesh(
            hist_by_time.index.seconds/60/60,
            hist_by_time.columns,
            hist_by_time.T,
            vmax=vmax,
            cmap=cmap,
            norm=plot_norm,
            rasterized=True,
        )

    else:
        print(hist_by_time.shape, hist_by_time.max().max())
        c = ax.pcolormesh(
            hist_by_time.index,
            hist_by_time.columns,
            hist_by_time.T,
            vmax=vmax,
            cmap=cmap,
            norm=plot_norm,
            rasterized=True,
        )

        # raise ValueError(f'unrecognized Time index type {index_type}')

    if not log_counts and cbar:
        cb = fig.colorbar(
            c,
            cmap=cmap,
            ax=ax,
            # cax = fig.add_axes([1.02, 0.152, 0.03, 0.7])
        )

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.ticklabel_format(style='sci', scilimits=(6, 6))
        cb.ax.yaxis.get_offset_text().set_position((0, 1.01))
        cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
        cb.ax.yaxis.get_offset_text().set_verticalalignment('bottom')

        cb.set_label(
            '#/bin',
            labelpad=-8,
            y=-0.12,
            # x=-1,
            rotation=0,
            va='bottom',
            ha='right'
        )

    elif cbar:
        cb = fig.colorbar(
            c,
            cmap=cmap,
            ax=ax,
            # cax = fig.add_axes([1.01, 0.125, 0.015, 0.85])
        )

        cb.set_label(
            '#/bin',
            labelpad=-8,
            y=-0.12,
            # x=-1,
            rotation=0,
            va='bottom',
            ha='right'
        )


    ax.set_ylabel(power_label)
    ax.set_title(title)

    # X axis formatting
    if issubclass(index_type, pd.Timestamp):
        xaxis_concise_dates(fig, ax)
    else:
        locator = ax.xaxis.get_major_locator()
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:0.0f}:00'))
        # draw()
        # labels = [f'{l.get_text()}:00' for l in ax.get_xticklabels()]
        # ax.set_xticklabels(labels)

    return ax

def plot_full_series(fc, stat_df):
    fc_map = dict(zip(np.round(stat_df.index.levels[0],1),stat_df.index.levels[0]))    
    fig, ax = subplots()

    # I just ran into that in the above plots too... a tacky workaround was to plot
    # each contiguous segment. See if this does what you were aiming for -Dan
    stat_df = stat_df.loc[fc_map[fc]]
    for seg in contiguous_segments(stat_df, 'Time'):
        seg.plot(ax=ax, ls='-.', color=c1[:len(seg.columns)], label=None, rasterized=True)
#         ax.plot(seg.reset_index().Time,seg.Power,'-.')
        
    ax.set_ylabel ()
#     plt.show()
    ax.legend(stat_df.columns, loc='best')
    
    return ax

def between(hist_df, start, end):
    df = hist_df.drop(columns='Frequency')
    df.reset_index(inplace=True)
    df.set_index('Time', inplace=True)
    df = df.between_time(start, end)
    df.reset_index(inplace=True)
    df.set_index(['Time'],inplace=True)
    return df

def between_by_day(hist_df,start,end):
    df = between(hist_df,start,end)
    df.reset_index(inplace=True)
    return df.groupby(df.Time.dt.date).mean()

def plot_by_day(fc,hist_df):
    ax = plt.gca()
    fc_map = dict(zip(np.round(hist_df.index.levels[0],1),hist_df.index.levels[0]))
    dwell_fc = hist_df.loc(axis=0)[fc_map[fc]]
    splits = {
    'Morning': between_by_day(dwell_fc, '0:00', '6:00'),
    'Afternoon': between_by_day(dwell_fc, '6:00', '12:00'),
    'Evening': between_by_day(dwell_fc, '12:00', '18:00'),
    'Night': between_by_day(dwell_fc, '18:00', '00:00')
    }
    time_label_list =  ['Morning','Afternoon','Evening','Night']
    for time_label in time_label_list:
        color = next(ax._get_lines.prop_cycler)['color']
        if len(splits[time_label]) > 0:
            plt.plot(splits[time_label].reset_index().Time,splits[time_label],'.',color=color, rasterized=True)
            plt.hlines(y=splits[time_label].mean(),xmin=splits[time_label].reset_index().Time[0],xmax=splits[time_label].reset_index().Time[len(splits[time_label])-1],colors=color)
    plt.title(f'{np.round(fc,1)} MHz')
    plt.ylabel ('Power averaged over each day')
    ax.legend(['Morning','Afternoon','Evening','Night','Average Morning','Average Avernoon','Average Evening','Average Night'])
    plt.show()

def plot_normed_histogram(hists_normed, fc, bounds_dB=(-40,60)):
    fig, ax = subplots()
    
    hists_normed = hists_normed.sum()[bounds_dB[0]:bounds_dB[1]]/hists_normed.sum().sum()
    
    hists_normed.plot(ax=ax, linestyle='', marker='.', logy=True, rasterized=True)
    
    xlabel(f'Power normed to {median_period} median (dB)')
    ylabel(f'Fraction of power samples');
    title(f'{np.round(fc,1)} MHz')

def plot_counts_each_frequency(hists, fc_list=None, xlim=(-70,70), title=None, ax=None, logy=True):
    if fc_list is not None:
        fc_map = dict(zip(np.round(hists.index.levels[0],1),hists.index.levels[0]))
        hists = hists.loc[[fc_map[fc] for fc in fc_list]]
    else:
        fc_list = np.round(hists.index.levels[0],1)

    ax = (hists
     .groupby('Frequency')  # perform computation at each Frequency
     .sum()                 # over all time
    ).T.plot(logy=logy, rasterized=True, ax=ax)

    ax.legend([f'{np.round(fc,1)} MHz' for fc in fc_list])
    ax.set_xlim(xlim)

    if not logy:
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_position((-.03, 1.01))
        ax.yaxis.get_offset_text().set_horizontalalignment('right')
        ax.yaxis.get_offset_text().set_verticalalignment('top')

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)        
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs = numpy.arange(1.0, 10.0) * 0.1, numticks = 10))

    ax.set_ylabel('Sample count')
    ax.set_xlabel(power_label)
    if title is not None:
        ax.set_title(title)

    return ax

def plot_survival_each_frequency(rcum_hists, fc_list=None, title=None, noise_survival=None, noise_margin_dB=6, xlim=None, figsize=None, ax=None, logy=True, norm='all'):
    if 'Site' in rcum_hists.index.names:
        site_count = len(rcum_hists.index.levels[rcum_hists.index.names.index('Site')])
    else:
        site_count = 1

    def do_site_norm(df):
        """ for groupby(['Frequency', 'Site]) - normalize by total number of samples in each site in each freq
        """
        return df/(df.iloc[:,0].sum()*site_count)

    if norm not in (None, False, 'all', 'site'):
        raise ValueError(f"norm argument must be one of (None, False, 'all', 'site')")

    if ax is None:
        fig, ax = subplots(figsize=figsize)

    if fc_list is not None:
        rcum_hists = histogram_analysis.loc_by_fc_spec(rcum_hists, fc_list)
    else:
        fc_list = np.round(rcum_hists.index.get_level_values('Frequency').unique(),1)

    row_counts = rcum_hists.iloc[:,0]
    counts_by_fc = row_counts.groupby('Frequency', sort=False).size()*SAMPLES_PER_HISTOGRAM

    # compute masks from threshold levels before we do any normalization or scaling
    if noise_survival is not None:
        mask_locs = (
            rcum_hists
            .groupby('Frequency', group_keys=True, sort=False)
            .apply(
                histogram_analysis.mask_survival_near_noise,
                noise_survival,
                noise_margin_dB
            )
        ).T.values

    else:
        mask_locs = np.zeros(shape=(rcum_hists.columns.size, len(fc_list)),dtype='bool')

    if norm == 'site':
        rcum_hists = rcum_hists.groupby(['Frequency', 'Site'], sort=False).apply(do_site_norm)

    survival = (
        rcum_hists
        .groupby('Frequency', sort=False)  # a computation at each Frequency
        .sum()                 # over time and sites
        .T
    ).astype('float')

    if norm == 'all':
        survival.values[:] = survival.values[:] / survival.values[0,:][np.newaxis,:]

    try:
        ax = survival.mask(mask_locs).plot(logy=logy, rasterized=True, ax=ax)
    except ValueError:
        print(survival.shape, mask_locs.shape)
        raise

    # reset and reuse the color cycle
    # ax.set_prop_cycle(None)
    # ax = survival.mask(~mask_locs.values).plot(logy=logy, rasterized=True, ax=ax, ls=(0,(1,10)))

    if logy:
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=11))
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs = numpy.arange(.2,1,.1), numticks=12))


    ax.set_xlim(xlim)    

    if norm in ('all', 'site'):
        ax.set_ylabel('Fraction of samples > band power')
        radix = int(np.floor(np.log10(counts_by_fc.mean())))
        mult = int(np.round(counts_by_fc.mean()/10**(radix)).mean())

        ax.set_title(
            f"""Empirical SF, """
            f"""{"averaged by site, " if norm == 'site' else ", "}"""
            f"""{"log scale, " if logy else "linear scale, "}"""            
            f"""{title}""",
            visible=False
        )

        N_text = f'Each $N \\approx {mult}\\times 10^{{{radix}}}$'
        if title is None:
            title = N_text
        else:
            title = f'{title}\n{N_text}'

        ax.legend([f'{np.round(fc,1)} MHz' for fc in fc_list], title=title)
    else:
        ax.set_title(
            f"""Empirical SF, unnormalized,"""
            f"""{"averaged by site, " if norm == 'site' else ", "}"""
            f"""{title}""",
            visible=False
        )
        ax.set_ylabel('Number of samples > band power')
        ax.legend([f'{np.round(fc,1):f} MHz' for fc in fc_list], title=title)
    ax.set_xlabel(power_label)

    if title is not None:
        caption =  f"""Empirical survival functions of band power in {title} plotted by center frequency. """
    else:
        caption =  f"""Empirical survival functions of band power plotted by center frequency. """
    if norm == 'all':
        caption += "Sample counts from each tester were summed together to compute the SF. "
    elif norm == 'site':
        caption += "The SF shown is the average of the SFs computed for each test site. "

    if logy:
        caption += "To illustrate rare events, the traces are transformed onto a logarithmic vertical axis. "
    else:
        caption += "The vertical scale is linear to emphasize the most frequent power levels. "

    # if noise_survival is not None:
    #     caption += "Dotted lines are shown at power levels at which noise or compression inside the sensor introduce at least 1\,dB of distortion."

    set_caption(gcf(), caption)
    
    return ax

def plot_survival_each_site(rcum_hists, title=None, noise_survival=None, noise_margin_dB=6, xlim=None, figsize=None, ax=None, logy=True, norm='all'):
    if norm not in (None, False, 'all'):
        raise ValueError(f"norm argument must be one of (None, False, 'all')")

    site_list = rcum_hists.index.get_level_values('Site').unique().sort_values()

    if ax is None:
        fig, ax = subplots(figsize=figsize)

    row_counts = rcum_hists.iloc[:,0]
    counts_by_site = row_counts.groupby('Site', sort=False).size()*SAMPLES_PER_HISTOGRAM

    # compute masks from threshold levels before we do any normalization or scaling
    if noise_survival is not None:
        mask_locs = (
            rcum_hists
            .groupby('Site', group_keys=True, sort=False)
            .apply(
                histogram_analysis.mask_survival_near_noise,
                noise_survival,
                noise_margin_dB
            )
        ).T

        mask_locs = (mask_locs.max(axis=1)[:,np.newaxis] + np.zeros((1,len(site_list)))).astype('bool')

    else:
        mask_locs = np.zeros(shape=(rcum_hists.columns.size, len(rcum_hists.index.levels[1])),dtype='bool')

    survival = (
        rcum_hists
        .groupby('Site', sort=False)  # a computation at each Frequency
        .sum()                 # over all time and sites
        .T
    ).astype('float')

    if norm == 'all':
        survival.values[:] = survival.values[:] / survival.values[0,:][np.newaxis,:]

    ax = survival.mask(mask_locs).plot(logy=logy, rasterized=True, ax=ax)

    if logy:
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=11))
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs = numpy.arange(.2,1,.1), numticks=12))
    # reset and reuse the color cycle
    # ax.set_prop_cycle(None)
    # ax = survival.mask(~mask_locs).plot(logy=logy, rasterized=True, ax=ax, ls=(0,(1,10)))

    ax.set_xlim(xlim)

    if norm == 'all':
        ax.set_ylabel('Fraction of samples > band power level')
        # radix = np.floor(np.log10(counts_by_site)).astype(int)
        # mult = np.round(counts_by_site/10**(radix)).astype(int)

        ax.set_title(
            f"""Empirical SF, """
            f"""{"log scale, " if logy else "linear scale, "}"""            
            f"""{title}""",
            visible=False
        )

        # N_text = f'$N \\approx {mult}\\times 10^{{{radix}}}$'
        # if title is None:
        #     title = N_text
        # else:
        #     title = f'{N_text}'

        ax.legend(site_list, title='Site', ncol=2)

        ylim([-.01,1.01])
    else:
        ax.set_title(
            f"""Empirical SF, unnormalized,"""
            f"""{title}""",
            visible=False
        )
        ax.set_ylabel('Number of samples > band power level')
        ax.legend(site_list, title=title, ncol=2)
    ax.set_xlabel(power_label)

    caption =  f"""Empirical survival functions of band power in {title} plotted by site. """
    if norm == 'all':
        caption += "Sample counts from each tester were summed together to compute the SF. "
    elif norm == 'site':
        caption += "The SF shown is the average of the SFs computed for each test site. "

    if logy:
        caption += "To illustrate rare events, the traces are transformed onto a logarithmic vertical axis. "
    else:
        caption += "The vertical scale is linear to emphasize the most frequent power levels. "

    if noise_survival is not None:
        caption += "Dotted lines are shown at power levels at which noise or compression inside the sensor introduce at least 1\,dB of distortion."

    set_caption(gcf(), caption)
    
    return ax

# def plot_survival_each_frequency_dwell_norm(hists, fc_list, title, median_period, dwell_stat='min', survival_kws):
#     h_normed_rcum, dwell = histogram_analysis.rcum_on_normalized_per_dwell(
#         hists,
#         fc_list,
#         title,
#         median_period,
#         dwell_stat='min',
#     ) 

#     ax = plot_survival_each_frequency(
#         h_normed_rcum,    
#         title=title,
#         fc_list=fc_list,
#         **survival_kws
#     );
#     ax.axhline(.9, ls=':', color='k')
#     ax.axhline(.1, ls=':', color='k')
#     ax.set_xlabel(f'Band power relative to power at dwell window histogram max (dB)')
#     set_caption(
#         ax.get_figure(),
#         f"""Empirical distribution of the received interference floor in LTE """
#         f"""uplink bands. Sample counts summed over all sites between """
#         f"""{" and ".join(histogram_analysis.date_span(hists))}. Dotted lines """
#         f"""indicate the 10th and 90th percentiles."""
#     )

def plot_heatmap_relative_to_daily_median(hists, fc_list):
    fig, axs = subplots(3,2, figsize=(figsize_fullwidth[0], 3*figsize_fullwidth[1]), sharey=True)

    axs = list(axs[0])+list(axs[1])+list(axs[2])

    for fc, ax in zip(fc_lte_dl, axs):
        h_fc = histogram_analysis.loc_by_fc_spec(hists, fc)

        daily_median = histogram_analysis.compute_dwell_median(
            h_fc,
            rolling_stat_period='1D',
            rolling_stat_name='median',
            by=['Site']
        )

        h_normed = histogram_analysis.normalize_power_per_dwell(h_fc, daily_median)

        # # mask out power levels near the noise floor
        # dl_average_power = daily_median.groupby(['Frequency','Site'], sort=False).apply(lambda df: 0*df+df.mean())
        # dl_average_power = dl_average_power.mask(dl_average_power < -75) # normalize only for values about 10 dB greater than the noise floor
        # dl_hists_normed = histogram_analysis.normalize_power_per_dwell(dl_hists, dl_average_power)

        h_tod = (
            histogram_analysis.rebin_time_scale(h_normed, '10T')
            .groupby('Time')
            .sum()
        )

        plot_power_histogram_heatmap(
            h_tod,
            title=f'{np.round(fc,1)} MHz',
            bounds=(-9,9),
            log_counts=False,
            ax=ax,
            cbar=(fc == fc_lte_dl[0])
        )

        ax.yaxis.get_label().set_visible(False)
        ax.xaxis.get_label().set_visible(False)

    fig.supxlabel('Time of day (local time)')
    fig.supylabel('Band power relative to daily median in each site (dB)')

    axs[-1].set_visible(False)

    fig.suptitle('LTE downlink histogram heat maps by time of day')
    set_caption(
        fig,
        f"""Histogram heat maps of samples received in each LTE downlink band by
        time of day. Sample counts are summed across all sites in {histogram_analysis.date_bounds(hists)}.
        """
    )

    return fig, axs

