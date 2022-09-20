#!/usr/bin/env python3

# Command line tool for manipulating the yaml files in COVID-19 spectrum monitoring data

import click
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import glob
import sys
from numbers import Number
import shutil

from src.read_dat import yaml_metadata
from src import munging
import config

COEFFICIENT_FIELDS = ("noise_average", "power_correction")
ORIGINALS_SUBDIR = "backup"

HELP = "manage calibration values in the .yaml files generated in a dataset of swept-power spectrum monitoring"


def read_cal_xlsx(path_to_xlsx):
    """ read an calibration xlsx file and return a dictionary with the 
        same format as yaml_metadata (without SDR acquisition parameters)
    """
    sheets = pd.read_excel(path_to_xlsx, sheet_name=None, engine="openpyxl")

    power_corrections = []
    noise_averages = []

    for fc_group in (700.0, 2400.0, 5500.0):
        # pull in workbook data by frequency
        i = sheets[f"iChannel_{int(fc_group)}MHz"].set_index("Frequency (GHz)")["Gain"]
        q = sheets[f"qChannel_{int(fc_group)}MHz"].set_index("Frequency (GHz)")["Gain"]
        power_corrections += [1.0 / (i + q)]

        i = sheets[f"iChannel_{int(fc_group)}MHz"].set_index("Frequency (GHz)")[
            "Noise (mW)"
        ]
        q = sheets[f"qChannel_{int(fc_group)}MHz"].set_index("Frequency (GHz)")[
            "Noise (mW)"
        ]

        # noise averages need to be averaged, because the noise power performed
        # separately on each of the I and Q channels refers the _entire_ input noise power
        # to that channel
        noise_averages += [0.5 * (i + q)]

    coefficients = pd.DataFrame(
        dict(
            power_correction=pd.concat(power_corrections, axis=0).sort_index(),
            noise_average=pd.concat(noise_averages, axis=0).sort_index(),
        )
    )

    coefficients.index = coefficients.index * 1e9  # to Hz
    coefficients.index.name = "Frequency (Hz)"

    return coefficients.T.to_dict()


def yaml_find_noise_backfills(tester_cals):
    """ Return a 2-column DataFrame with columns ('From', 'Into') representing
        pairs of noise copy operations needed to backfill missing noise values.
    """
    nocal = (tester_cals != 0).all(axis=1) == False

    # a dataframe with the name of the yamls
    yamls = pd.DataFrame(
        2
        * [
            [
                p.with_name(p.name[: -len(".".join(p.suffixes)) + 1] + ".yaml")
                for p in tester_cals.index
            ]
        ],
        columns=tester_cals.index,
        index=["From", "Into"],
    ).T

    # blank 'From' where there is no cal
    yamls.loc[nocal, "From"] = None

    # backfill file names for each of the blanked 'From'
    return yamls.bfill(axis=0).loc[nocal]


def backup_yaml(path):
    """ If a backup of the untouched yaml file has not been created, create one.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    if not path.name.endswith(".yaml"):
        raise ValueError(f"{path} does not have .yaml file extension")

    # back up the original yaml files
    original = path.parent / ORIGINALS_SUBDIR / path.name
    if not original.exists():
        print(f"- backing up {path} -> {original}")
        (path.parent / ORIGINALS_SUBDIR).mkdir(exist_ok=True)
        shutil.copy2(path, original)


def copy_coefficients(src, dest, field, force=False, write=False):
    """
        :src: path to the source yaml file, or already-loaded metadata
        :dest: path to the destination yaml file
        :field: one of 'power_correction' or 'noise_average'
        
        :returns: True if the fields in dest needed to be copied, False if they are the same
    """

    if field not in COEFFICIENT_FIELDS:
        raise KeyError(f"field '{field}' is not one of '{COEFFICIENT_FIELDS}'")

    if isinstance(src, dict):
        metadata_from = src
    else:
        metadata_from = yaml_metadata(src)

    metadata_into = yaml_metadata(dest)
    
    if not force:
        for k in metadata_into.keys():
            if isinstance(k, Number):            
                if metadata_from[k][field] != metadata_into[k][field]:
                    break
        else:
            return False
        
    if not write:
        # at this point we know whether we need to 
        return True

    for k in metadata_into.keys():
        if isinstance(k, Number):
            if k in metadata_from:
                metadata_into[k][field] = metadata_from[k][field]
            else:
                print(f"warning - field {k} not in source, skipping")

    del metadata_into["center_frequencies"]

    backup_yaml(dest)
    with open(dest, "w") as f:
        yaml.dump(metadata_into, stream=f, Dumper=yaml.dumper.SafeDumper)
        print(f"- wrote calibration data to {dest}")
        
    return True


def expand_wildcards(paths):
    """ expand wildcards in each path in `paths`
    """
    expanded = []
    for p in paths:
        if "*" in p:
            expanded += glob.glob(p)
        else:
            expanded.append(p)

    return expanded


@click.group(help=HELP)
# @click.pass_context
def cli():
    pass


@cli.command(
    name="backfill-noise", help="backfill missing noise calibration coefficients"
)
@click.argument(
    "data-root", default=config.DEFAULT_DATA_ROOT, type=click.Path(exists=True)
)
@click.option(
    "--write", is_flag=True, help="write the proposed changes to the paths DESTS"
)
def noise_backfill(DATA_ROOT, write):
    # load the calibration data from all of the metadata
    noise_average = []
    testers = []

    # pull in the calibration data
    print(f"finding swept data in {DATA_ROOT}")
    paths = munging.find_swept_power_monitoring(DATA_ROOT)

    for p in paths:
        metadata = yaml_metadata(p)
        noise_average.append(
            {k: metadata[k]["noise_average"] for k in metadata["center_frequencies"]}
        )
        testers.append(p.parent.name)
    noise_average = pd.DataFrame(noise_average, index=paths)
    noise_average["Tester"] = testers

    # generate a table of needed yaml file replacements
    replacements = noise_average.groupby("Tester").apply(yaml_find_noise_backfills)

    print(
        f"{len(replacements)} of {len(noise_average)} yaml files are missing noise calibration data\n"
    )
    for from_, into in replacements.values:
        if from_ is None:
            print(
                f"noise calibration data needed to backfill {into}, but none is available"
            )
            continue

        if copy_coefficients(from_, into, "noise_average", write=write):
            print(f"'noise_average' {from_} -> {into}")

    if not write:
        sys.stderr.write(f"to commit these operations to disk, run again with --write")


@cli.command(
    name="copy-noise", help="copy noise readings from one .yaml to one or more others"
)
@click.argument("src", nargs=1, type=click.Path(exists=True, dir_okay=False))
@click.argument("dests", nargs=-1, type=click.Path())
@click.option(
    "--write", is_flag=True, help="write the proposed changes to the paths DESTS"
)
def noise_copy(src, dests, write):
    if src.endswith(".yaml"):
        metadata_from = yaml_metadata(src)
    elif src.endswith(".xlsx"):
        metadata_from = read_cal_xlsx(src)
    else:
        raise ValueError(f"source file {src} has an invalid file extension")

    for dest in expand_wildcards(dests):
        if copy_coefficients(metadata_from, dest, "noise_average", write=write):
            print(f"'noise_average' {src} -> {dest}")

    if not write:
        sys.stderr.write(f"to commit these operations to disk, run again with --write")


@cli.command(
    name="reset", help="restore .yaml noise readings to original reported values"
)
@click.argument(
    "list of yaml files", nargs=-1, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--write", is_flag=True, help="write the proposed changes")
def reset(dests, write):
    for dest in dests:
        if not dest.name.endswith("yaml"):
            raise ValueError(f"can only reset yaml files, not {dest}")

        backup_path = dest.parent / ORIGINAL_SUBDIR / dest.name

        if not backup_path.exists():
            sys.stderr.write(f"no original exists for {dest}, skipping")
            continue
        shutil.copy2(backup_path, dest)
    raise NotImplementedError


@cli.command(
    name="copy-power",
    help="copy power corrections from one .yaml or .xlsx cal data file into one or more others",
)
@click.argument("src", nargs=1, type=click.Path(exists=True, dir_okay=False))
@click.argument("dests", nargs=-1, type=click.Path())
@click.option(
    "--write", is_flag=True, help="write the proposed changes to the paths DESTS"
)
def power_copy(src, dests, write):
    if src.endswith(".yaml"):
        metadata_from = read.yaml_metadata(src)
    elif src.endswith(".xlsx"):
        metadata_from = read_cal_xlsx(src)
    else:
        raise ValueError(f"source file {src} has an invalid file extension")

    for dest in expand_wildcards(dests):
        if copy_coefficients(metadata_from, dest, "power_correction", write=write):
            print(f"'power_correction' {src} -> {dest}")

    if not write:
        sys.stderr.write(f"to commit these operations to disk, run again with --write")


@cli.command(name="power-reset", help="restore power gain to their GNURadio defaults")
@click.argument("list of yaml files", nargs=-1, type=click.Path())
@click.option("--write", is_flag=True, help="write the proposed changes")
def power_reset(dests, write):
    raise NotImplementedError


if __name__ == "__main__":
    cli()
