## COVID-19 Analysis README

### Installation and environment setup
This environment, unlike the acquisition, assumes you have anaconda python distribution (python >= 3.7).

1.	Since certain core packages need to be up to date, it's best to do these with the anaconda package manager. If you haven't updated since before June 2020, update with
	```
		conda update --all
	```
2.	Install the remaining python requirements and make sure that pyppyteer (which is used for pdf generation) is set up

	```
		pip install -r REQUIREMENTS.txt
        pyppeteer-install
	```

### General analysis procedure
The swept-frequency spectrum monitoring dataset files needs to be accessible in a single directory. The
default location of this directory is set in `config.py` as `DEFAULT_DATA_ROOT`, which we refer to here
as `config.DEFAULT_DATA_ROOT`. The value of `config.DEFAULT_DATA_ROOT` in the repository is the
default windows path to the google drive file stream, but this can also be overridden in the command line.

The general procedure to run the complete analysis and report generation is as follows:

1. Digest the raw dataset into parquet files.
    ```
    python analyze.py
    ```
2. Generate an overview report of the entire dataset.
    ```
    python report.py reports\summary.ipynb
    ```
3. Generate separate reports for each unique combination of (tester, day) in the dataset.
    ```
    python report.py reports\daily_by_tester.ipynb
    ```

Calibration data management is provided by `yamltool.py`. It reads .yaml metadata and
and .xlsx calibration data, and broadcasts noise or power level calibration information into one or
more .yaml files.

Detailed configuration for the behavior of these command line operations is
given in `config.py`. Discussion about the motivation behind these parameter
values is commented there.

### Calibration data management

1. To apply calibration values to the yaml files in the dataset
    ```
    python .\yamltool.py copy-power --write "G:\Shared drives\COVID-19 Spectrum Monitoring\Calibration\MUF Calibration Results With Uncertainty\SDR_Calibration_Data.xlsx" "G:\Shared drives\COVID-19 Spectrum Monitoring\Data\*\*.yaml"
    ```