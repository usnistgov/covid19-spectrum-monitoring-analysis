import click
from pathlib import Path
import pandas as pd
import numpy as np
import read

DEFAULT_DATA_ROOT = r"G:\Shared drives\COVID-19 Spectrum Monitoring\Calibration"


def convert_directory(DATA_ROOT):
    DATA_ROOT = Path(DATA_ROOT)
    for path in Path(DATA_ROOT).rglob("*.swept_power.dat"):
        data, metadata = read.swept_average(path, holdoff=0.195)

        dwell_average = (
            data.mean(axis=1)
            .reset_index()
            .pivot(index="Sweep", columns="Frequency", values=0)
        )

        dest = path.with_name(
            ".".join(path.name.split(".")[:-2]) + ".dwell_average.csv"
        )
        dwell_average.to_csv(dest)
        print(f"wrote {dest}")


@click.command()
@click.argument("DATA_ROOT", required=False)
def run(DATA_ROOT):
    if DATA_ROOT is None:
        DATA_ROOT = DEFAULT_DATA_ROOT

    convert_directory(DATA_ROOT)


if __name__ == "__main__":
    run()
