# routines for reading spectrum monitoring data files

import numpy as np
import pandas as pd
import os.path
import yaml
from pathlib import Path


def _prepend_index_levels(df, **label_value_map):
    """ prepend levels named by label_value_map.keys() with 
        constant values from label_value_map.values().
    """
    new_code = np.zeros(df.index.codes[0].size)
    
    levels = [pd.Index([v]) for k,v in label_value_map.items()] + df.index.levels
    codes = (len(label_value_map)*[new_code]) + df.index.codes
    names = list(label_value_map.keys()) + df.index.names

    return pd.DataFrame(
        df.values,
        columns=df.columns,
        index=pd.MultiIndex(levels=levels, codes=codes, names=names)
    )


def yaml_metadata(data_path: str) -> dict:
    """load a yaml metadata into a dict"""
    entries = os.path.split(data_path)
    name = os.path.splitext(entries[-1])[0].rsplit(".", 1)[0] + ".yaml"
    yaml_path = os.path.join(*(entries[:-1] + (name,)))
    if not os.path.exists(yaml_path):
        raise IOError('expected metadata file "%s" does not exist' % yaml_path)
    with open(yaml_path, "rb") as f:
        try:
            metadata = yaml.load(f, Loader=yaml.SafeLoader)
        except OSError as ex:
            raise IOError(f"failed to read {data_path}") from ex
            
    if metadata is None:
        raise IOError(f"{str(data_path)} is empty")

    metadata["center_frequencies"] = sorted(
        [k for k in metadata.keys() if isinstance(k, float)]
    )
    return metadata


def time_span(path):
    file_size = path.stat().st_size

    metadata = yaml_metadata(str(path))
    dwell_fields = int((metadata["dwell_time"]) / metadata["aperture_time"] + 5)
    sweep_size = len(metadata["center_frequencies"]) * dwell_fields * 4

    t0 = np.fromfile(str(path), offset=4, count=1, dtype="float64")[0]

    t1 = np.fromfile(
        str(path),
        offset=4 + ((file_size // sweep_size) - 1) * (sweep_size),
        count=1,
        dtype="float64",
    )[0]

    return t0, t1


def subset(span_test, span_ref):
    """ returns True if the time span `span_test` (as returned by `time_span`)
        is entirely within the other time span `span_ref`
    """
    return span_test[0] >= span_ref[0] and span_test[1] <= span_ref[1]


def _assert_version(expected, actual):
    if actual != expected:
        raise ValueError(
            "expected file format code %i., but given %f" % (expected, actual)
        )


def valid_swept_average_path(path, required_extensions):
    """ If the path refers to a file among a valid data set,
        return the path to the .swept_power.dat file. Otherwise, return None.
    """
    path = Path(path)

    if path.is_dir():
        return None

    # join all suffixes to catch from the first '.'
    suffix = "".join(path.suffixes).lower()

    # only care about .yaml and .dat files
    if suffix not in [ext.lower() for ext in required_extensions]:
        return None

    # ensure all required companion files are present
    for ext in set(required_extensions):
        companion = Path(str(path)[: -len(suffix)] + ext)
        if not companion.exists():
            logger.info(
                f"new {path.relative_to(DATA_ROOT)}, "
                f"awaiting {companion.relative_to(DATA_ROOT)}"
            )
            return None
        if ext == ".swept_power.dat":
            if companion.stat().st_size == 0:
                logger.info(f"{companion} is empty!")
                return None
            if companion.stat().st_size < 1504 * 21 * 4:
                logger.info(f"{companion} has less than 1 sweep")
                return None

    datfile = Path(str(path)[: -len(suffix)] + ".swept_power.dat")
    version = int(-np.fromfile(datfile, dtype="float32", count=1))
    if version not in (3, 4):
        logger.warning(f"unsupported file version {version} in {path}")
        return None

    return Path(str(path)[: -len(suffix)] + ".swept_power.dat")


def swept_average_merge(src_path, dest_path):
    # pull in metadata
    src_metadata = yaml_metadata(src_path)
    dest_metadata = yaml_metadata(dest_path)

    dwell_samples = int(src_metadata["dwell_time"] / src_metadata["aperture_time"])
    dwell_count = len(src_metadata["center_frequencies"])
    field_count = 5 + dwell_samples
    sweep_fields = field_count * dwell_count

    entries = os.path.split(src_path)
    name = os.path.splitext(entries[-1])[0].rsplit(".", 1)[0] + ".yaml"
    yaml_path = os.path.join(*(entries[:-1] + (name,)))

    # validate metadata
    missing = set(dest_metadata.keys()).difference(src_metadata.keys())
    if len(missing) > 0:
        raise KeyError("dest file is missing metadata keys %s" % str(missing))
    mismatched = {
        k for k in dest_metadata.keys() if src_metadata[k] != dest_metadata[k]
    }
    if len(mismatched) > 0:
        raise ValueError("dest file metadata mismatched for keys %s" % str(mismatched))

    src = np.fromfile(src_path, dtype="float32")
    dest = np.fromfile(dest_path, dtype="float32")

    if src.size > 0 and src[0] != -4:
        raise ValueError("both file versions must be 4, but source is v%i" % (-src[0]))

    if dest.size > 0:
        if dest[0] > -4:
            raise ValueError(
                "both file versions must be 4, but destination version is v%i"
                % (-dest[0])
            )

        # truncate dest to an integer number of sweeps
        dest = dest[: (dest.shape[0] // sweep_fields) * sweep_fields]

    # merge the data and write
    merged = np.append(dest, src)
    merged.tofile(dest_path)
    os.remove(src_path)
    os.remove(yaml_path)


from scipy import special


def estimate_X(Y, n, sigma_N):
    """ Estimate the average input signal power, X, based on
        signal power reading Y and 
    """
    ynorm = Y * n / sigma_N ** 2
    return Y * special.gammainc(n, ynorm) - sigma_N ** 2 * special.gammainc(
        n + 1, ynorm
    )


def estimate_var_X(Y, n, sigma_N):
    """ Estimate the average input signal power, X, based on
        signal power reading Y and 
    """
    z = Y * n / sigma_N ** 2
    f = special.gammainc

    return (
        (
            n ** 2 * f(n, z) * f(n + 1, z) ** 2
            - 2 * n ** 2 * f(n + 1, z) ** 2
            + n ** 2 * f(n + 2, z)
            - 2 * n * z * f(n, z) ** 2 * f(n + 1, z)
            + 4 * n * z * f(n, z) * f(n + 1, z)
            - 2 * n * z * f(n + 1, z)
            + n * f(n + 2, z)
            + z ** 2 * f(n, z) ** 3
            - 2 * z ** 2 * f(n, z) ** 2
            + z ** 2 * f(n, z)
        )
        * sigma_N ** 4
        / n ** 2
    )


def _swept_power_raw(path: str):
    """ Read a .swept_power.dat file and return it as a pandas.DataFrame instance.
        The file must report version 4.
    """
    metadata = yaml_metadata(path)
    dwell_time = metadata["dwell_time"]
    aperture_time = metadata["aperture_time"]
    center_frequencies = metadata["center_frequencies"]

    dwell_samples = int(round(dwell_time / aperture_time))

    raw = np.fromfile(path, dtype="float32")

    # Older file format versions are not supported.
    _assert_version(-4, raw[0])
    metadata["version"] = -4

    #     print('reading %s (v4 file)'%path)

    # each dwell window consists of (version_tag, 2-field timestamp, frequency, dwell_window_sample_peak,)
    # then the sequence of dwell_samples samples of average power
    field_count = 5 + dwell_samples

    # truncate to an integer number of dwell windows
    raw = raw[: (raw.size // field_count) * (field_count)]  # only complete fields
    return raw, metadata

import labbench as lb

def swept_power(
    path: str,
    holdoff: float = 0.05,
    dwell_settle: float = 0,
    calibrate: bool = True,
    excess_noise_subtraction_dB: float = 0.0,
    dof_factor: float = 1,  # multiply chi-squared distribution dof by this factor
    subtract_ambient: ("power", "magnitude", "distribution", False) = "power",
):
    """ Read a .swept_power.dat file and return it as a pandas.DataFrame instance.
        The file must report version 4.
    """
    metadata = yaml_metadata(path)
    dwell_time = metadata["dwell_time"]
    aperture_time = metadata["aperture_time"]
    center_frequencies = metadata["center_frequencies"]

    dwell_samples = int(round(dwell_time / aperture_time))
    throwaway = int(round(holdoff / aperture_time))

    raw = np.fromfile(path, dtype="float32")

    # Older file format versions are not supported.
    _assert_version(-4, raw[0])
    metadata["version"] = -4

    #     print('reading %s (v4 file)'%path)

    # each dwell window consists of (version_tag, 2-field timestamp, frequency, dwell_window_sample_peak,)
    # then the sequence of dwell_samples samples of average power
    field_count = 5 + dwell_samples

    # truncate to an integer number of dwell windows
    raw = raw[: (raw.size // field_count) * (field_count)]  # only complete fields
    raw = raw.reshape(raw.size // field_count, field_count)
    raw = raw[: (raw.shape[0] // len(center_frequencies)) * len(center_frequencies)]

    # compute the index values
    timestamps = np.frombuffer(raw[:, 1:3].tobytes(), dtype="float64") * 1e9

    # an acquisition bug causes timestamps not to be updated when subsequent
    # samples are taken at the same frequency. correct this here with an estimated timestamp
    timestamps[np.where(np.diff(timestamps, prepend=np.nan) == 0)] += (metadata['dwell_time'])*1e9

    timestamps = (
        pd.DatetimeIndex(timestamps.astype("datetime64[ns]"))
        .tz_localize("utc")
        .tz_convert("America/Denver")
    )
    frequency = np.round(raw[:, 3] * (10/1e6)) / 10
    sweep = np.floor(np.arange(raw.shape[0]) / len(center_frequencies)).astype(int)

    # generate DataFrames
    index = pd.MultiIndex.from_arrays(
        [sweep, timestamps, frequency], names=["Sweep", "Time", "Frequency"]
    )

    metadata["peak_power"] = pd.DataFrame(
        raw[:, 4:5], columns=["Sample peak"], index=index
    )

    spectrum = pd.DataFrame(
        raw[:, 5 + throwaway :],
        columns=np.arange(throwaway, dwell_samples) * metadata["aperture_time"],
        index=index,
    )

    spectrum.columns.name = "Dwell time elapsed (s)"

    # the actual hardware synchronization is loose. invalid and nan values may intermittently
    # continue for up to a couple hundred ms on some SDRs. this fills in everything through
    # the final nan value with nan. later code will interpret this data to ignore.
    #     correction = np.cumsum(spectrum.shift(throwaway, axis=1).values[:,::-1],axis=1)[:,::-1]*0
    #     spectrum.values[:] += correction # set nans
    # #     spectrum.values[:,:dwell_settle] = np.nan
    #     spectrum = spectrum.dropna(axis=1, how='all')

    # apply the calibration data
    if calibrate:
        fc_inds = spectrum.index.get_level_values('Frequency')
        fc_unique = np.array(spectrum.index.levels[2])

        # map to the rounded frequency in the metadata file
        fc_lookup = (np.round(fc_unique * 10) / 10.0) * 1e6

        power_corrections = np.array([metadata[fc]['power_correction'] for fc in fc_lookup])
        noise_averages = np.array([metadata[fc]['noise_average'] for fc in fc_lookup])

        _, cal_inds = np.where(fc_inds.values[:,np.newaxis] == fc_unique[np.newaxis,:])

        if subtract_ambient == "power":
            spectrum.values[:] -= noise_averages[cal_inds, np.newaxis]

        elif subtract_ambient == 'magnitude':
            spectrum.values[:] = (np.sqrt(spectrum.values)-np.sqrt(noise_averages[cal_inds, np.newaxis]))**2

        elif subtract_ambient == "distribution":
            sigma_N = np.sqrt(metadata[fc_lookup]["noise_average"])
            n = metadata["sample_rate"] * metadata["aperture_time"] * dof_factor
            spectrum.values[:] = estimate_X(spectrum.values, n=n, sigma_N=sigma_N)

        spectrum.values[:] *= power_corrections[cal_inds, np.newaxis]

    return spectrum, metadata


def _swept_power_v1(path, holdoff=0.325, dwell_settle=0, calibrate=True):
    metadata = yaml_metadata(path)
    dwell_time = metadata["dwell_time"]
    aperture_time = metadata["aperture_time"]
    center_frequencies = metadata["center_frequencies"]
    dwell_samples = int(round(dwell_time / aperture_time))

    raw = np.fromfile(path, dtype="float32")
    _assert_version(-1, raw[0])
    metadata["version"] = -1

    # each dwell window consists of (version_tag, frequency, dwell_window_sample_peak,)
    # then the sequence of dwell_samples samples of average power
    field_count = 3 + dwell_samples

    # truncate to an integer number of dwell windows
    raw = raw[: (raw.size // field_count) * (field_count)]

    # fix for frequency corruption that happened in some cases
    #     raw[1::field_count] = (raw[1::field_count].size//len(center_frequencies))*center_frequencies

    spectrum = pd.DataFrame(raw.reshape(raw.size // field_count, field_count))
    spectrum.reset_index(inplace=True)

    del raw
    spectrum.columns = ["Time", "Version", "Frequency", "Sample peak"] + list(
        range(dwell_samples)
    )
    spectrum["Frequency"] = np.round(spectrum["Frequency"] / 1e6 * 10) / 10
    frequency_count = spectrum.Frequency.unique().size

    # spectrum.Time = dwell_time*frequency_count*np.floor(spectrum.Time/frequency_count)
    # spectrum.Frequency /= 1e6
    # spectrum['Frequency'] = np.round(spectrum['Frequency']*10)/10 # round to nearest 1 kHz
    spectrum.set_index(["Time", "Frequency"], inplace=True)

    # move the sample peaks to metadata, and remove it from spectrum
    metadata["peak_sample"], spectrum = spectrum.iloc[:, 1:2], spectrum.iloc[:, 2:]
    metadata["peak_sample"].reset_index(inplace=True)
    metadata["peak_sample"].set_index("Time",inplace=True)

    # Ettus USRPs output 0 as an "invalid" sentinel
    # spectrum = spectrum.replace(0,np.nan)

    # the actual hardware synchronization is loose. invalid and nan values may intermittently
    # continue for up to a couple hundred ms. this fills in everything through the final
    # nan value with nan. later code will interpret this data to ignore.
    throwaway = int(round(holdoff / aperture_time))
    correction = (
        np.cumsum(spectrum.shift(throwaway, axis=1).values[:, ::-1], axis=1)[:, ::-1]
        * 0
    )
    spectrum.values[:] += correction
    spectrum.values[:, :dwell_settle] = np.nan
    spectrum = spectrum.dropna(axis=1, how="all")

    # repeat this, because bizarrely there is rounding error otherwise
    spectrum.reset_index(inplace=True)
    spectrum["Frequency"] = np.round(spectrum["Frequency"] * 10) / 10
    spectrum["Time"] = pd.Timestamp.fromtimestamp(
        metadata["start_timestamp"]
    ) + pd.to_timedelta(spectrum.Time, unit="s")
    spectrum["Sweep"] = np.floor(
        np.arange(spectrum.shape[0]) / len(center_frequencies)
    ).astype(int)
    spectrum.set_index(["Sweep", "Time", "Frequency"], inplace=True)
    spectrum.columns.name = "Aperture power sample"

    if calibrate:
        for fc in spectrum.index.levels[2]:
            if fc <= 0.0:
                raise ValueError(
                    f"record at invalid frequency {fc} in '{path}'; all frequencies: {spectrum.index.levels[2]}"
                )
            idx = np.where(spectrum.index.get_level_values(2) == fc)
            fc_lookup = (np.round(fc * 10) / 10.0) * 1e6

            if subtract_ambient == "power":
                spectrum.values[idx] += (
                    -metadata[fc_lookup]["noise_average"] * excess_noise_factor
                )
            elif subtract_ambient == "magnitude":
                spectrum.values[idx] = (
                    np.sqrt(spectrum.values[idx])
                    - np.sqrt(metadata[fc_lookup]["noise_average"])
                ) ** 2
            elif subtract_ambient == "distribution":
                sigma_N = np.sqrt(metadata[fc_lookup]["noise_average"])
                n = metadata["sample_rate"] * metadata["aperture_time"] * dof_factor
                spectrum.values[idx] = estimate_X(
                    spectrum.values[idx], n=n, sigma_N=sigma_N
                )
            elif subtract_ambient != False:
                raise ValueError(
                    f"subtract_ambient argument must be one of ('power','magnitude',False), not {repr(subtract_ambient)}"
                )

            spectrum.values[idx] *= metadata[fc_lookup]["power_correction"]

    return spectrum, metadata


def _swept_power_v0(path, holdoff=0.18, dwell_settle=0):
    metadata = yaml_metadata(path)
    dwell_time = metadata["dwell_time"]
    aperture_time = metadata["aperture_time"]
    center_frequencies = metadata["center_frequencies"]
    dwell_samples = int(round(dwell_time / aperture_time))

    raw = np.fromfile(path, dtype="float32")
    raw = raw[: (raw.size // (dwell_samples)) * dwell_samples]

    global spectrum  # for debug
    spectrum = pd.DataFrame(
        raw.reshape(raw.size // dwell_samples, dwell_samples)
    ).reset_index()
    del raw
    spectrum.columns = ["Time"] + list(np.arange(dwell_samples))
    frequency_count = len(center_frequencies)  # spectrum.Frequency.unique().size

    spectrum["Time"] = (
        dwell_time * frequency_count * np.floor(spectrum.Time / frequency_count)
    )

    spectrum["Frequency"] = (
        list(center_frequencies)
        * int(np.ceil(spectrum.shape[0] / len(center_frequencies)))
    )[: spectrum.shape[0]]
    spectrum.Frequency /= 1e6
    spectrum.loc[:, "Frequency"] = (
        np.round(spectrum.loc[:, "Frequency"] * 10) / 10.0
    )  # round to nearest 1 kHz
    spectrum = spectrum.set_index(["Time", "Frequency"])

    # Ettus USRPs output 0 as an "invalid" sentinel
    spectrum = spectrum.replace(
        0, np.nan
    )  # specific to Ettus USRPs? They seem to give '0' until

    # the actual hardware synchronization is loose. invalid and nan values may intermittently
    # continue for up to a couple hundred ms. this fills in everything through the final
    # nan value with nan. later code will interpret this data to ignore.
    throwaway = int(round(holdoff / aperture_time))
    correction = (
        np.cumsum(spectrum.shift(throwaway, axis=1).values[:, ::-1], axis=1)[:, ::-1]
        * 0
    )
    spectrum.values[:] += correction
    spectrum.values[:, :dwell_settle] = np.nan

    spectrum = spectrum.reset_index()
    spectrum["Frequency"] = np.round(spectrum["Frequency"] * 10) / 10
    spectrum["Time"] = pd.Timestamp.fromtimestamp(
        metadata["start_timestamp"]
    ) + pd.to_timedelta(spectrum.Time, unit="s")
    spectrum["Sweep"] = np.floor(
        np.arange(spectrum.shape[0]) / len(center_frequencies)
    ).astype(int)
    spectrum = spectrum.set_index(["Sweep", "Time", "Frequency"])
    spectrum.columns.name = "Aperture power sample"

    return spectrum, metadata


# def swept_average(path, holdoff=None, dwell_settle=0, **kws):
#     version = -np.fromfile(path, dtype='float32', count=1)

#     if version.size == 0:
#         raise IOError('empty data file')
#     else:
#         version = version[0]

#     kws = dict(kws, path=path, holdoff=holdoff, dwell_settle=dwell_settle)
#     if holdoff is None:
#         kws.pop('holdoff')

#     if version == 4:
#         return _swept_average_v4(**kws)

#     else:
#         raise ValueError('data file reports unsupported version number "%f"'%version)


def swept_iq(path, holdoff=None, dwell_settle=0, **kws):
    version = -np.fromfile(path, dtype="float32", count=1)

    if version.size == 0:
        raise IOError("empty data file")
    else:
        version = version[0]

    kws = dict(kws, path=path, holdoff=holdoff, dwell_settle=dwell_settle)
    if holdoff is None:
        kws.pop("holdoff")

    elif version == 4:
        data, metadata = _swept_average_v4(**kws)
        data = pd.DataFrame(
            data.values[:, ::2] + data.values[:, 1::2] * 1j,
            index=data.index,
            columns=data.columns[::2],
        )

        return data, metadata

    else:
        raise ValueError('data file reports unsupported version number "%f"' % version)

        
def _greedy_swept_power(path, holdoff=0.195, subtract_ambient='power', calibrate=True,skip_sweeps=0):
    try:
        s,m = swept_power(
            path,
            calibrate=calibrate, 
            subtract_ambient=subtract_ambient,
            holdoff=holdoff
        )
        
        if skip_sweeps:
            s= s.iloc(axis=0)[skip_sweeps:]
        
        return s,m
    
    except IndexError:
        return None, None

def read_site_map(root):
    root = Path(root)

    return (
        pd
        .read_csv(root/'site_hash_mapping.csv')
        .set_index('Site')
        .iloc[:,0]
        .to_dict()
    )

def swept_power_glob(root, pattern='Data/*/*cold noise*.dat', site_map=None, **swept_power_kws):
    root = Path(root)

    dfs = []

    if site_map is None:
        site_map = read_site_map(root)
    elif site_map is False:
        site_map = {} 
    
    for p in root.glob(pattern):
        df = _greedy_swept_power(p, **swept_power_kws)[0]
        
        if df is None:
            continue
        
        df = _prepend_index_levels(
            df,
            Site=site_map.get(p.parent.name, p.parent.name),
            Path=p.with_suffix('').name
        )
        
        dfs.append(df)

    return pd.concat(dfs, copy=False).reorder_levels(['Site','Path','Frequency','Time','Sweep'])

def swept_power_single_frequency(power_path, holdoff=0):
    spectrum, metadata = swept_power(power_path, holdoff=holdoff)
    power = spectrum.droplevel(['Sweep','Frequency'])

    # convert time units to time elapsed
    power.index = (power.index - power.index[0]).total_seconds()

    if power.index[-1] == 0:
        power.index += metadata['dwell_time']*np.arange(power.index.size)

    power = power.unstack()
    power.index = power.index.get_level_values(0)+power.index.get_level_values(1)
    power = power.sort_index()
    
    power.name = 'Power (dBm/10 MHz)'

#     power = pd.Series(
#         0.5*(power.values[::2]+power.values[1::2]),
#         index=np.round(power.index[::2] - power.index[0],3),
#         name='Power'
#     ) # average into 1 ms bins
    
    return power, spectrum, metadata
