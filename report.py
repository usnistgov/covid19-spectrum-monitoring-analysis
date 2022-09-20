import click
import config
import templatepress  # , filesummary, globalsummary, logger

# import dask
from pathlib import Path
import labbench as lb
import pandas as pd
from traceback import format_exc
import os
from functools import partial
import dask
import numpy as np
import time
import contextlib
import pickle

default_target_root = Path(config.EXPORT_DATA_ROOT)/'Analysis'


def enumerate_nbparams(data_root, nb_func):
    """ return a list of dicts keyed on the subset of ('site','year','month','day'),
        used in the notebook for the Paperpress instance `press`,
        given the dataset in `data_root`.
    """
    global timestamps

    empty = pd.read_parquet(
        Path(data_root) / "histogram.parquet",
        columns=[],
        use_legacy_dataset=False,
        use_threads=False,
        memory_map=True        
    )
    sites = empty.index.levels[empty.index.names.index("Site")]

    valid = {}
    for t in sites:
        timestamps = empty.loc(axis=0)[:, t].index.get_level_values("Time")

        for date in set(timestamps.date):
            complete = dict(site=t, year=date.year, month=date.month, day=date.day,)

            filtered = {
                k: v for k, v in complete.items() if k in nb_func.default_parameters
            }

            filtered['DATA_ROOT'] = data_root

            # keying on the list of values ensures only unique values
            valid[tuple(filtered.values())] = filtered

    return list(valid.values())


def output_path(path, target_root, kwargs, suffix=".pdf"):
    """ Determine the subdirectory and file names of reports
    autogenerated from the noteboook at `path`.
    
    :path: path to the notebook
    :target_root: the root directory for autogenerated reports
    :kwargs: dictionary of the parameters set by the notebook
    """
    nb_name = Path(path).with_suffix("").name

    date_fields = [kwargs[k] for k in ("year", "month", "day") if k in kwargs]
    datestamp = "-".join([f"{field:02}" for field in date_fields])

    field_labels = []

    if "site" in kwargs:
        field_labels += ['site_'+kwargs['site']]

    if len(date_fields) > 0:
        field_labels += [f'from_{datestamp}']

    if len(field_labels) > 0:
        path = target_root/f'{nb_name}/{"-".join(field_labels)}{suffix}'
    else:
        path = target_root/f'{nb_name}{suffix}'

    return str(path)

def press_notebook(path, data_root, target_root, single=False, export_data=False):
    """ Make a pdf report using the jupyter notebook
        at `path`, the root directory of input data `data_root`,
        and the root directory for output reports, `target_root`.
    """
    target_root = Path(target_root)

    def single_press(kws):
        pdf_path = output_path(path, target_root, kws)

        with templatepress.NotebookFunction(path, suppress_exc=True) as call_notebook:
            nb_path = call_notebook(figure_format=config.FIGURE_FORMAT, **kws)
            templatepress.export_pdf(nb_path, pdf_path)

    nbparams = enumerate_nbparams(data_root, templatepress.NotebookFunction(path))

    if export_data:
        for kws in nbparams:
            kws['EXPORT_DATA_ROOT'] = str(default_target_root/'..')

    if single:
        single_press(nbparams[0])

    else:
        templatepress.logger.info(f"queuing jobs")
        with dask.config.set(scheduler="processes"), lb.stopwatch("report generation"):
            pending = [dask.delayed(single_press(kws)) for kws in nbparams]

            templatepress.logger.info(f"pressing {path} for {len(nbparams)} reports")
            dask.compute(pending, num_workers=int(os.cpu_count()))

    return


@click.command(
    help="generate a report using the jupyter notebook at PATH (or each notebook listed in a text file at PATH that lists relative paths to .ipynb files) from parquet files produced by analysis.py"
)
@click.argument("path", required=False, type=click.Path(exists=True))
@click.option(
    "--data-root",
    help=f"directory that contains the parquet data file inputs",
    default=config.DEFAULT_DATA_ROOT,
    type=click.Path(exists=True),
    show_default=True,
)
@click.option(
    "--target-root",
    default=default_target_root,
    type=click.Path(),
    show_default=True,
    help=f"destination root directory for the pdf report",
)
@click.option("--single", is_flag=True, type=bool)
@click.option("--export-data", is_flag=True, help='export data from the notebook, if supported')
def run(path, data_root, target_root, single=False, export_data=False):
    # path = Path(path)

    target_root.mkdir(exist_ok=True, parents=True)

    if path is not None:
        press_notebook(path, data_root, target_root, single=single, export_data=export_data)
    else:
        for p in config.ALL_REPORTS:
            press_notebook(p, data_root, target_root, single=single, export_data=export_data)


if __name__ == "__main__":
    run()
