#!/usr/bin/env python3
import numpy as np
from strf.rfio import Spectrogram

import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots()
    ax.imshow(s.z, origin="lower", aspect="auto", interpolation="None",
              vmin=vmin, vmax=vmax)

    plt.show()
