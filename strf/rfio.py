#!/usr/bin/env python3
import os
import re
import sys
import glob

import h5py
import json

import numpy as np

from datetime import datetime, timedelta

class Spectrogram:
    """Spectrogram class"""

    def __init__(self, t, fcen, bw, freq, z, siteid, location, tle):
        self.z = z
        self.nchan = self.z.shape[0]
        self.nsub = self.z.shape[1]
        self.t = t
        self.fcen = fcen
        self.bw = bw
        self.freq = freq
        self.siteid = siteid
        self.location = location
        self.tle = tle

    @classmethod
    def from_spectrogram(cls, froot, ifile, nsub, siteid):
        """Define a spectrogram"""
        path, prefix = extract_path_and_prefix(froot)

        # Read matching filenames
        fnames = sorted(glob.glob(os.path.join(path, f"{prefix}_*.bin")))

        # Start filename
        fname = os.path.join(path, f"{prefix}_{ifile:06d}.bin")

        if not fname in fnames:
            # Exit on no matching files
            if fnames == []:
                print(f"Spectrogram is not available under {fname}")
                sys.exit(1)
            else:
                print(f"Spectrogram is not available under {fname}\nUsing {fnames[0]} instead")
                fname = fnames[0]
                ifile = int(fname.split("_")[-1].replace(".bin", ""))

        # Read first header
        with open(fname, "rb") as fp:
            header = parse_header(fp.read(256))

        # Set frequencies
        freq_cen, bw, nchan = header["freq"], header["bw"], header["nchan"]
        freq_min, freq_max = -0.5 * bw, 0.5 * bw
        freq = freq_cen + np.linspace(freq_min, freq_max, nchan, endpoint=False) + freq_max / nchan
        
        # Loop over subints and files
        zs = []
        t = []
        isub = 0;
        while isub<nsub:
            # File name of file
            fname = os.path.join(path, f"{prefix}_{ifile:06d}.bin")

            # Exit on absent file
            if not os.path.exists(fname):
                break

            print(f"Opened {fname}")
            with open(fname, "rb") as fp:
                next_header = fp.read(256)
                while next_header:
                    header = parse_header(next_header)
                    t.append(header["utc_start"] + timedelta(seconds=0.5 * header["length"]))
                    z = np.fromfile(fp, dtype=np.float32, count=nchan)
                    # Break on incomplete spectrum
                    if len(z) != nchan:
                        break
                    zs.append(z)
                    next_header = fp.read(256)
                    isub += 1
            ifile += 1

        z = np.transpose(np.vstack(zs))
            
        print(f"Read spectrogram\n{nchan} channels, {nsub} subints\nFrequency: {freq_cen * 1e-6:g} MHz\nBandwidth: {bw * 1e-6:g} MHz")

        return cls(t, freq_cen, bw, freq, z, siteid, None, None)


    @classmethod
    def from_artifact(cls, filename):
        hdf5_file = h5py.File(filename, "r")

        if hdf5_file.attrs["artifact_version"] != 2:
            raise Exception(hdf5_file.attrs["artifact_version"])

        wf = hdf5_file.get("waterfall")
        metadata = json.loads(hdf5_file.attrs["metadata"])
        start_time = datetime.strptime(wf.attrs["start_time"].decode("ascii"), "%Y-%m-%dT%H:%M:%S.%fZ")
        z = np.transpose((np.array(wf["data"]) * np.array(wf["scale"]) + np.array(wf["offset"])))
        fcen = float(metadata["frequency"])
        freq = np.array(hdf5_file.get("waterfall").get("frequency")) + fcen
        bw = (freq[1] - freq[0]) * z.shape[0]
        t = [ start_time + timedelta(seconds=x) for x in  np.array(hdf5_file.get("waterfall").get("relative_time"))]
        location = metadata["location"]
        tle = [ x for x in metadata["tle"].split("\n") if x.strip() != "" ]
        nsub, nchan = z.shape
        
        print(f"Read artifact\n{nchan} channels, {nsub} subints\nFrequency: {fcen * 1e-6:g} MHz\nBandwidth: {bw * 1e-6:g} MHz")
        
        return cls(t, fcen, bw, freq, z, None, location, tle)
        
def extract_path_and_prefix(fname):
    basename, dirname = os.path.basename(fname), os.path.dirname(fname)

    pattern_with_extension = "_\d{6}.bin$"
    pattern_without_extension = "_\d{6}$"    
    if re.findall(pattern_with_extension, basename):
        prefix, _ = re.split(pattern_with_extension, basename)
    elif re.findall(pattern_without_extension, basename):
        prefix, _ = re.split(pattern_without_extension, basename)
    else:
        prefix = basename

    return dirname, prefix
        
def parse_header(header_b):
    header_s = header_b.decode("ASCII").strip("\x00")
    regex = r"^HEADER\nUTC_START    (.*)\nFREQ         (.*) Hz\nBW           (.*) Hz\nLENGTH       (.*) s\nNCHAN        (.*)\nNSUB         (.*)\nEND\n$"
    try:
        match = re.fullmatch(regex, header_s, re.MULTILINE)
    except AttributeError:
        match = re.match(regex, header_s, flags=re.MULTILINE)

    utc_start = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S.%f")

    return {"utc_start": utc_start,
            "freq": float(match.group(2)),
            "bw": float(match.group(3)),
            "length": float(match.group(4)),
            "nchan": int(match.group(5)),
            "nsub": int(match.group(6))}
    

def get_site_info(fname, site_id):
    with open(fname, "r") as fp:
        lines = fp.readlines()

    sites = []
    for line in lines:
        if "#" in line:
            continue
        parts = line.split()
        try:
            site = {"no": int(parts[0]),
                    "lat": float(parts[2]),
                    "lon": float(parts[3]),
                    "height": float(parts[4]),
                    "observer": " ".join(parts[5:])}
        except:
            print(f"Failed to read site {line}")

        sites.append(site)

    for site in sites:
        if site["no"] == site_id:
            return site

    return None
        
def get_frequency_info(fname, fcen, fmin, fmax):
    with open(fname, "r") as fp:
        lines = fp.readlines()
    C = 299792.458 # km/s
    padding = fcen * 20 / C # padding in MHz
    fmin_padding,fmax_padding = (fmin - padding) * 1e-6, (fmax + padding) * 1e-6
    frequencies = {}
    for line in lines:
        if "#" in line:
            continue
        parts = line.split()
        try:
            freq = { "norad": int(parts[0]), "freq": float(parts[1])}
        except:
            print(f"Failed to read frequency {line}")
        
        if freq["freq"] >= fmin_padding and freq["freq"] <= fmax_padding:
            if freq["norad"] in frequencies:
                frequencies[freq["norad"]].append(freq["freq"])
            else:
                frequencies[freq["norad"]] = [ freq["freq"] ]
    
    return frequencies


def  get_satellite_info(fname, frequencies):
    with open(fname, "r") as fp:
        lines = [x.strip() for x in fp.readlines()]
    
    sat_freq = []
    for i in range(0, len(lines), 3):
        noradid = int(lines[i+2].split()[1])
        if noradid in frequencies:
            entry = { "noradid" : noradid, "frequencies": frequencies[noradid], "tle": lines[i:i+3] }
            sat_freq.append(entry)
    
    return sat_freq
