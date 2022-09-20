# -*- coding: utf-8 -*-
"""
Produce pdf reports from jupyter notebook templates via papermill and wxhtmltopdf

TODO: Split this into a separate library?

July 2020
Dan Kuester 
"""

__all__ = ["NotebookFunction", "logger", "export_pdf"]

# first take care of path for pdfkit
import os
from pathlib import Path
from traceback import format_exc

from xml.dom import minidom


# Monkeypatch pyppeteer to increase timeout, which is needed to accommodate possible latency 
import pyppeteer
import pyppeteer.errors

import notebook_as_pdf
import nbconvert

from labbench import retry
import logging, logging.config
from nbclient.exceptions import CellTimeoutError
import papermill

from tempfile import TemporaryDirectory
import time
from functools import wraps, partial
from traitlets.config import get_config
import zipfile

# lengthen the timeout leash for PDF export
class _Page(pyppeteer.page.Page):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self.setDefaultNavigationTimeout(60000)
pyppeteer.page.Page, pyppeteer.page._Page = _Page, pyppeteer.page.Page

def _setup_logger(suppress=[]):
    info_fmt = "%(asctime)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    logger = logging.getLogger("paperpress-")
    logging.basicConfig(
        level=logging.INFO, format=info_fmt, datefmt=date_fmt
    )
    logging.basicConfig(filename=f"{__name__}.log", level=logging.DEBUG)

    ch = logging.FileHandler("paperpress.log")
    ch.setFormatter(logging.Formatter(info_fmt, datefmt=date_fmt))
    logger.addHandler(ch)
    
    for name in suppress:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        
    return logger

@wraps(papermill.execute_notebook)
@retry(CellTimeoutError, tries=5)
def execute_notebook(*args, **kws):
    return papermill.execute_notebook(*args, **kws)

logger = _setup_logger(suppress=("blib2to3.pgen2.driver", "papermill", "traitlets"))

class NotebookFunction:
    def __init__(self, nb_path:Path, suppress_exc=False):
        """
            :template_path: path to the jupyter notebook to run 
            :nb_parameters: parameter defaults for calling the ipynb
        """
        self.template_path = Path(nb_path).absolute()

        # pull in notebook parameters from the notebook and nb_parameters
        self._temp_root = None
        self.default_parameters = self._get_nb_parameters()
        self.suppress_exc = suppress_exc

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        self.close()

    def open(self):
        if self._temp_root is not None:
            raise IOError(f'already opened a copy of the notebook')
        self._temp_root = TemporaryDirectory(prefix="paperpress")

    def close(self):
        if self._temp_root is not None:
            self._temp_root.cleanup()
            self._temp_root = None

    def _get_nb_parameters(self):
        """ return a dictionary of the default parameters defined in the notebook
            at `path`.
        """

        with self:
            nb = execute_notebook(
                str(self.template_path),
                str(Path(self._temp_root.name) / "parse.ipynb"),  #
                prepare_only=True,
                timeout=600,
            )

        # pick out parameter cells
        cells = [c for c in nb["cells"] if "parameters" in c["metadata"]["tags"]]
        if len(cells) == 0:
            raise IOError("no cells tagged with parameters!")

        # run and stash results in local
        locals_ = {}
        for c in cells:
            exec(c["source"], {}, locals_)
            
        return locals_

    def _validate_parameter_names(self, parameters):
        """ raise exception for parameters that are not defined in the notebook
        """
        invalid_parameters = set(parameters).difference(self.default_parameters)
        if len(invalid_parameters) > 0:
            raise KeyError(
                f"the parameter variable(s) {tuple(invalid_parameters)} "
                f"are not defined in {str(self.template_path)}"
            )

    def __call__(self, **kws):
        """ Executes a copy the notebook in a temporary directory with parameters set by the given keyword arguments.
        Returns a path to a copy of the notebook after it is run
        """
        self._validate_parameter_names(kws)

        if self._temp_root is None:
            raise IOError(f"{self.template_path} needs to be opened to call")

        temp_root = Path(self._temp_root.name)

        temp_nb = temp_root / self.template_path.name

        t0 = time.perf_counter()

        # run the notebook at the specified path
        logger.debug(f"\trun {self.template_path}")

        kws = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in kws.items()
        }

        try:
            execute_notebook(
                str(self.template_path),
                str(temp_nb),
                parameters=dict(self.default_parameters, **kws),
                cwd=str(self.template_path.parent),
            )

        except KeyboardInterrupt:
            raise

        except:
            text = format_exc()
            logger.error(text)
            name = self.template_path.with_suffix("").name
            with open(f"{name}-errors.log", "a") as log:
                log.write(
                    f'\n\n{time.asctime()}: {self.template_path}\n{"-"*80}\n{text}\n'
                )

            if not self.suppress_exc:
                raise
                
        logger.info(
            f'ran "{self.template_path}" in {time.perf_counter()-t0:0.1f} s'
        )
                
        return temp_nb


def _get_svg_info(svg_contents):
    def get_info_field(n, *names):
        try:
            n = n.getElementsByTagName(names[0])[0]
        except IndexError:
            print('no stuff for name ', names[0])
            return ''

        if len(names) > 1:
            return get_info_field(n, *names[1:])

        if n is not None and n.firstChild is not None:
            return n.firstChild.data
        else:
            return ''

    node = minidom.parseString(svg_contents)
    node = node.getElementsByTagName('svg')[0]

    return get_info_field(node, 'title'), get_info_field(node, 'metadata', 'rdf:RDF', 'cc:Work', 'dc:date')

def get_figure_metadata(contents: bytes):
    title_, date = _get_svg_info(contents)
    title_, *caption = (title_ or 'untitled').split('##', 1)

    if len(caption) > 0:
        caption = caption[0]
    else:
        caption = ""

    return title_, caption, date



def _sanitize_filename(s):
    return (
        s
        .replace(':', '_')
        .replace('/', '_')
        .replace('\\', '_')
        .replace('*', '')
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('__', '_')
    )


def export_pdf(notebook_path, output_path):
    @retry((OSError, pyppeteer.errors.NetworkError), tries=5)
    def press_pdf(exporter, nb):
        """ wrapper retries on certain errors
        """
        return exporter.from_filename(nb)
    
    conf = get_config()
    conf.NotebookClient.iopub_timeout = 15
    conf.NbConvertApp.export_format = "pdftohtml"
    conf.TemplateExporter.exclude_code_cell = False
    conf.TemplateExporter.exclude_input = True
    conf.TemplateExporter.exclude_input_prompt = True
    conf.TemplateExporter.exclude_markdown = False
    conf.TemplateExporter.exclude_output = False
    conf.TemplateExporter.exclude_output_prompt = True
    conf.TemplateExporter.exclude_raw = False
    conf.TemplateExporter.exclude_unknown = False
    conf.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.coalesce_streams",
        "nbconvert.preprocessors.CSSHTMLHeaderPreprocessor",
    ]
    conf.ExtractOutputPreprocessor.enabled = True
    conf.TemplateExporter.filters = {
        'word_wrap': nbconvert.filters.wrap_text
    }
    conf.HTMLExporter.theme = 'light'
    
    output_path = Path(output_path)
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    t0 = time.perf_counter()
    exporter = notebook_as_pdf.PDFExporter(config=conf)
    pdf_contents, metadata = press_pdf(exporter, str(notebook_path))

    with open(output_path, 'wb') as fd:
        fd.write(pdf_contents)

    # save a copy of each figure
    if isinstance(metadata.get('outputs', None), dict):
        logger.warning(f'found outputs, time to try saving {len(metadata["outputs"])} figures')

        (output_path.parent/'figures').mkdir(exist_ok=True, parents=True)

        fig_nums = {}

        for key, contents in metadata['outputs'].items():
            if key.endswith('svg'):
                # a '##' string separates title from caption
                title_, caption, date = get_figure_metadata(contents)                

                fig_nums[title_] = fig_nums.get(title_, -1) + 1

                if fig_nums[title_] > 0:
                    fig_suffix = f'{title_}_{fig_nums[title_]:02d}.svg'
                else:
                    fig_suffix = f'{title_}.svg'

            else:
                fig_suffix = '_'.join(key.split('_',1)[1:])

            name = f'{output_path.stem}_{fig_suffix}'

            with open(output_path.parent/'figures'/name, 'wb') as fd:
                fd.write(contents)

    logger.info(
        f'exported report to "{output_path}" in {time.perf_counter()-t0:0.1f} s'
    )