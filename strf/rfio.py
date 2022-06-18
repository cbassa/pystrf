#!/usr/bin/env python3
import os
import re

from datetime import datetime

import numpy as np

class Spectrogram:
    """Spectrogram class"""

    def __init__(self, path, prefix, ifile, nsub, siteid):
        """Define a spectrogram"""

        # Read first file to get number of channels
        fname = os.path.join(path, "{:s}_{:06d}.bin".format(prefix, ifile))
        with open(fname, "rb") as fp:
            header = parse_header(fp.read(256))

        # Set frequencies
        freq = np.linspace(-0.5*header["bw"], 0.5*header["bw"], header["nchan"], endpoint=False)+header["freq"]+0.5*header["bw"]/header["nchan"]
        
        # Loop over subints and files
        zs = []
        mjds = []
        isub = 0;
        while isub<nsub:
            # File name of file
            fname = os.path.join(path, "{:s}_{:06d}.bin".format(prefix, ifile))
            with open(fname, "rb") as fp:
                next_header = fp.read(256)
                while next_header:
                    header = parse_header(next_header)
#                    mjds.append(Time(header["utc_start"], format="datetime", scale="utc").mjd+0.5*header["length"]/86400.0)
                    zs.append(np.fromfile(fp, dtype=np.float32, count=header["nchan"]))
                    next_header = fp.read(256)
                    isub += 1
            ifile += 1

        self.z = np.transpose(np.vstack(zs))
        self.mjd = np.array(mjds)
        self.freq = freq
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
    
