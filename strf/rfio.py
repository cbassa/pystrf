#!/usr/bin/env python3
import os
import re

import numpy as np

from datetime import datetime, timedelta

class Spectrogram:
    """Spectrogram class"""

    def __init__(self, path, prefix, ifile, nsub, siteid):
        """Define a spectrogram"""

        # Read first file to get number of channels
        fname = os.path.join(path, f"{prefix}_{ifile:06d}.bin")
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
            with open(fname, "rb") as fp:
                next_header = fp.read(256)
                while next_header:
                    header = parse_header(next_header)
                    t.append(header["utc_start"] + timedelta(seconds=0.5 * header["length"]))
                    zs.append(np.fromfile(fp, dtype=np.float32, count=nchan))
                    next_header = fp.read(256)
                    isub += 1
            ifile += 1

        self.z = np.transpose(np.vstack(zs))
        self.t = t
        self.freq = freq
        self.fcen = freq_cen
        self.siteid = siteid
        self.nchan = self.z.shape[0]
        self.nsub = self.z.shape[1]          


def parse_header(header_b):
    header_s = header_b.decode('ASCII').strip('\x00')
    regex = r"^HEADER\nUTC_START    (.*)\nFREQ         (.*) Hz\nBW           (.*) Hz\nLENGTH       (.*) s\nNCHAN        (.*)\nNSUB         (.*)\nEND\n$"
    try:
        match = re.fullmatch(regex, header_s, re.MULTILINE)
    except AttributeError:
        match = re.match(regex, header_s, flags=re.MULTILINE)

    utc_start = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S.%f')

    return {'utc_start': utc_start,
            'freq': float(match.group(2)),
            'bw': float(match.group(3)),
            'length': float(match.group(4)),
            'nchan': int(match.group(5)),
            'nsub': int(match.group(6))}
    

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
