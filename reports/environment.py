import warnings
warnings.simplefilter("ignore")

# covid spectrum monitoring read
import sys
sys.path.insert(1, "../src")
import read_dat

# Set up the plotting environment for notebooks that convert cleanly
# to pdf or html output.
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot

np = numpy
plt = pyplot

import IPython
import IPython.display
from IPython.display import display, HTML, set_matplotlib_formats
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *

# remaining imports
from importlib import reload
import ipywidgets as widgets
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import logging

import functools

_captions = {}

@functools.wraps(IPython.display.set_matplotlib_formats)
def set_matplotlib_formats(formats, *args, **kws):
    """ apply wrappers to inject title (from figure or axis titles) and caption (from set_caption metadata),
        when available, into image 'Title' metadata
    """

    IPython.display.set_matplotlib_formats(formats, *args, **kws)

    # monkeypatch IPython's internal print_figure to include title metadata
    from IPython.core import pylabtools
    pylabtools = reload(pylabtools)

    def guess_title(fig):
        if fig._suptitle is not None:
            return fig._suptitle.get_text()

        for ax in fig.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        else:
            return 'untitled'

    def title_to_label(title_):
        """ replace 1 or more non-alphanumeric characters with '-'
        """
        import re, string

        pattern = re.compile(r'[\W_]+')
        return pattern.sub('-', title_).lower()

    @functools.wraps(pylabtools.print_figure)
    def wrapper(fig=None, *a, **k):
        # a fresh mapping
        if fig is None:
            fig = pylab.gcf()

        k = dict(k)
        label = title_to_label(guess_title(fig))
        caption_text = _captions.get(id(fig), "")
        k.setdefault('metadata', {})['Title'] = f"{label}##{caption_text}" if caption_text else label

        ret = pylabtools._print_figure(fig, *a, **k)
        ext = formats if isinstance(formats, str) else formats[0]
        display(HTML(f'<tt>{label}.{ext}:</tt>{"<br>"+caption_text if caption_text else " (no caption data)"}'))
        return ret

    pylabtools.print_figure, pylabtools._print_figure = wrapper, pylabtools.print_figure


# requires pandas >= 1.0.0
convert_datetime = matplotlib.units.registry[np.datetime64]

def set_caption(fig, text):
    """ Associate caption text as metadata for the figure.
    """
    global _captions

    _captions[id(fig)] = text

# Alternative colorblind sets
# _Krzywinski15complementary = np.array([
#     '#68023F', '#00463C', '#008169', '#C00B6F', '#EF0096', '#00A090', '#00DCB5', '#FF95BA',
#     '#FFCFE2', '#5FFFDE', '#003C86', '#590A87', '#9400E6', '#0063E5', '#009FFA', '#ED0DFD',
#     '#FF71FD', '#00C7F9', '#7CFFFA', '#FFD5FD', '#6A0213', '#3D3C04', '#008607', '#C80B2A',
#     '#F60239', '#00A51C', '#00E307', '#FFA035', '#FFDC3D', '#9BFF2D'
# ]).reshape((15,2)).T

# _Krzywinski24complementary = np.array([
# '#003D30', '#5A0A33', '#005745', '#810D49', '#00735C', '#AB0D61', '#009175', '#D80D7B',
# '#00AF8E', '#FF2E95', '#00CBA7', '#FF78AD', '#00EBC1', '#FFACC6', '#86FFDE', '#FFD7E1',
# '#00306F', '#460B70', '#00489E', '#6B069F', '#005FCC', '#8E06CD', '#0079FA', '#B40AFC',
# '#009FFA', '#ED0DFD', '#00C2F9', '#FF66FD', '#00E5F8', '#FFA3FC', '#7CFFFA', '#FFD5FD',
# '#004002', '#5F0914', '#005A01', '#86081C', '#007702', '#B20725', '#009503', '#DE0D2E',
# '#00B408', '#FF4235', '#00D302', '#FF8735', '#00F407', '#FFB935', '#AFFF2A', '#FFE239',
# ]).reshape((24,2)).T

# krzywinski_colorblind_15 = mpl.colors.ListedColormap(_Krzywinski15complementary[0], name='krzywinski colorblind 15')
# krzywinski_colorblind_15_alt = mpl.colors.ListedColormap(_Krzywinski15complementary[1], name='krzywinski colorblind 15 (alt)')
# krzywinski_colorblind_24 = mpl.colors.ListedColormap(_Krzywinski24complementary[0], name='krzywinski colorblind 24')
# krzywinski_colorblind_24_alt = mpl.colors.ListedColormap(_Krzywinski24complementary[1], name='krzywinski colorblind 24 (alt)')

time_format = "%Y-%m-%d %H:%M:%S"


def ts(s):
    return pd.Timestamp(s, tz="America/Denver")

fc_lte_ul = [701.5, 709, 782, 821.3, 842.5]
fc_lte_dl = [734, 739, 751, 866.3, 887.5]
fc_quiet = 2695.
fc_ism = [2412., 2437., 2462.]
fc_unii = [5170., 5190.,5210., 5230., 5240., 5775., 5795.]

FC_NAMED = {
    'LTE uplink bands': fc_lte_ul,
    'LTE downlink bands': fc_lte_dl,
    'ISM band': fc_ism,
    'U-NII bands': fc_unii,
    'Quiet band': [fc_quiet],
}


# what IPython uses to display in the browser.
# IMPORTANT: pass rasterize=True in calls to plot, or the notebooks get very large!

# plot settings mostly designed for IEEE publication styles
sns.set(context="paper", style="ticks", font_scale=1)

def _set_colors_from_dict(color_dict, linestyle_cycle=None):
    from cycler import cycler 
    if linestyle_cycle is not None:
        cyc = cycler(linestyle=linestyle_cycle) * cycler(color=color_dict)
    else:
        cyc = cycler(color=color_dict)

    rc(
        'axes',
        prop_cycle = cyc 
    )

    rc(
        'patch',
        facecolor = list(color_dict.keys())[0]
    )

    for color, code in color_dict.items():
        # map the colors into color codes like 'k', 'b', 'c', etc
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb

# colorblind-accessible palette from http://mkweb.bcgsc.ca/colorblind/palettes.mhtml#page-container    
# _set_colors_from_dict(
#     {
#         '#2271B2': 'b',
#         '#3DB7E9': 'c',
#         '#359B73': 'g',
#         '#f0e442': 'y',
#         '#e69f00': 'o',        
#         '#F748A5': 'm',
#         '#d55e00': 'r',
#         '#000000': 'k',
#     },
#     ['-', '--', '-.', ':']
# ) 
sns.set_palette('colorblind', 7, color_codes=True)
rc('axes', prop_cycle=cycler(linestyle=['-', ':', '--']) * rcParams['axes.prop_cycle'])


# concise date formatting by default
converter = matplotlib.dates.ConciseDateConverter()
matplotlib.units.registry[np.datetime64] = converter
matplotlib.units.registry[datetime.date] = converter
matplotlib.units.registry[datetime.datetime] = converter

rc(
    "font",
    family=["serif"],
    serif=["Times New Roman"],
    weight="normal",
    size=10,
    cursive="Freestyle Script",
)

# support for TeX-like math expressions
# (without the slowdown of usetex=True)
rc(
    "mathtext",
    fontset="custom",
    it="serif:italic",
    rm="serif:normal",
    bf="serif:bold",
    default="it",
)

rc(
    "axes",
    labelweight="regular",
    **{"spines.top": False, "spines.right": False}
)

# tighten up saved figures for publication
rc(
    "savefig",
    bbox="standard",
    pad_inches=0,
    facecolor="none",  # equivalent to prior frameon=False
    #    transparent=False
)

# tighten up the legend
rc(
    'legend',
    handletextpad=0.2,
    labelspacing=.005,
    borderaxespad=0.05,
    columnspacing=0.5,
    handlelength=1.25,
    edgecolor='k'    
)

rc(
    "lines",
    linewidth=0.5
)

figsize_fullwidth = np.array([6.5,2.15])
figsize_halfwidth = np.array([3.2,2.15])

rc(
    "figure",
    figsize=figsize_halfwidth,  # autolayout=False,
    titlesize=10,
    dpi=300,
    **{
        "constrained_layout.use": True,
        "constrained_layout.h_pad": 0,
        "constrained_layout.w_pad": 0,
        #                 'subplot.left'    : 0.,  # the left side of the subplots of the figure
        #                 'subplot.right'   : 1.,    # the right side of the subplots of the figure
        #                 'subplot.bottom'  : 0.11,    # the bottom of the subplots of the figure
        #                 'subplot.top'     : 0.88,   # the top of the subplots of the figure
        #                 'subplot.wspace':  0,
        #                 'subplot.hspace': 0,
    }
)

rc(
    "svg",
    fonttype='none'
)

font = matplotlib.font_manager.findfont(
    matplotlib.font_manager.FontProperties(family=["serif"])
)
