#!/usr/bin/env python3
import numpy as np
from strf.rfio import Spectrogram

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


if __name__ == "__main__":
    # Settings
    path = "data"
    prefix = "2021-08-04T20:48:35"
    ifile = 50
    nsub = 1800

    # Read spectrogram
    s = Spectrogram(path, prefix, ifile, nsub, 4171)

    # Create plot
    vmin, vmax = np.percentile(s.z, (5, 99.95))

    # Time limits
    tmin, tmax = mdates.date2num(s.t[0]), mdates.date2num(s.t[-1])

    # Frequency limits
    fcen = np.mean(s.freq)
    fmin, fmax = (s.freq[0] - fcen) * 1e-6, (s.freq[-1] - fcen) * 1e-6
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(s.z, origin="lower", aspect="auto", interpolation="None",
              vmin=vmin, vmax=vmax,
              extent=[tmin, tmax, fmin, fmax])

    ax.xaxis_date()
    date_format = mdates.DateFormatter("%F\n%H:%M:%S")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"Frequency (MHz) - {fcen * 1e-6:g} MHz")

    
    plt.show()
