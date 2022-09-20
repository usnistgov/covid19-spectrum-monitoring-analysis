# Munging, validation, and bounds checking for swept_power files
#

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from traceback import format_exc
from dask import bag

try:
    from . import read_dat
except:
    import read_dat

# import logging
# logger = logging.getLogger('munging')

__all__ = ["find_swept_power_monitoring", "modified_time", "contains_monitoring"]

def glob_single(path, pattern):
    """ returns a single filein path matching pattern, or raises an exception """
    matches = tuple(Path(path).glob(pattern))
    if len(matches) == 0:
        raise IOError(f'no such file or directory "{path}/{pattern}"')
    if len(matches) > 1:
        raise IOError(f'multiple matches for "{path}/{pattern}"')
    return matches[0]


## Path munging
def modified_time(path):
    stat = Path(path).stat()
    t = max(stat.st_mtime, stat.st_ctime)

    if path.name.endswith(".swept_power.dat"):
        stat = (
            Path(path).with_name(".".join(path.name.split(".")[:-2]) + ".yaml").stat()
        )
        t = max(t, stat.st_mtime, stat.st_ctime)

    return t


def find_swept_power_monitoring(DATA_ROOT, min_sweeps=1):
    """ return a list of swept power data files matching `DATA_ROOT/*/*.swept_power.dat`
        which contain monitoring data (and not calibration data).
    """

    # all of the .swept_power.dat files
    candidates = sorted(Path(DATA_ROOT).glob("*/*.swept_power.dat"))

    # multiprocess evaluation of whether each file contains power monitoring data
    path_bag = bag.from_sequence(candidates)

    # empty the log file
    with open("ignored files.txt", "w") as fd:
        pass

    checked_paths = path_bag.map(lambda p: contains_monitoring(p, min_sweeps)).compute()

    # filter out the failed checks
    return [p for p in checked_paths if p is not None]


def contains_monitoring(path, min_sweeps=1):
    """ Check whether the file referenced by `path` is spectrum monitoring data
        as in the expected .swept_power.dat file, with at least `min_sweeps` full
        cycles through its frequency sweep list.
        
        :returns: `path` if it looks like monitoring data, otherwise None 
    """

    with open("ignored files.txt", "a") as fd:
        path = Path(path)

        if path.is_dir():
            return None

        # join all suffixes to catch from the first '.'
        suffix = "".join(path.suffixes).lower()

        # only care about .yaml and .dat files
        if suffix not in [ext.lower() for ext in (".yaml", ".swept_power.dat")]:
            fd.write(f"{path} has wrong extension\n")
            return None

        if "cold noise" in path.name:
            fd.write(f"{path} is calibration data\n")
            return None

        # ensure all required companion files are present
        for ext in (".swept_power.dat", ".yaml"):
            companion = Path(str(path)[: -len(suffix)] + ext)
            if not companion.exists():
                fd.write(f"{path} is missing companion data expected at {companion}\n")
                return None

        # minimum size checking
        datfile = Path(str(path)[: -len(suffix)] + ".swept_power.dat")
        metadata = read_dat.yaml_metadata(
            path
        )  # Path(str(path)[:-len(suffix)]+self.metadata_extension))

        dwell_fields = (metadata["dwell_time"]) / metadata["aperture_time"] + 4
        minimum_size = (
            2 * min_sweeps * len(metadata["center_frequencies"]) * dwell_fields * 4
        )

        if datfile.stat().st_size < minimum_size:
            fd.write(f"{path} has too little data\n")
            return None

        # only support version 4 for now, since there is little test data for <= 3
        version = int(-np.fromfile(datfile, dtype="float32", count=1))
        if version != 4:
            fd.write(
                f"{path} format version is {version} but the analysis supports only version 4\n"
            )
            return None

    return Path(str(path)[: -len(suffix)] + ".swept_power.dat")
