#!/usr/bin/env python3
import sys
import numpy as np
from strf.rfio import Spectrogram

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets  import RectangleSelector
import matplotlib as mpl
mpl.rcParams['keymap.save'].remove('s')
mpl.rcParams['keymap.fullscreen'].remove('f')

if __name__ == "__main__":
    # Settings
    path = "data"
    prefix = "2021-08-04T20_48_35"
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

    mark = ax.scatter([], [],c="white",s=5)
    line_fitting = ax.scatter([], [], edgecolors="yellow",s=10, facecolors='none')
    ax.imshow(s.z, origin="lower", aspect="auto", interpolation="None",
              vmin=vmin, vmax=vmax,
              extent=[tmin, tmax, fmin, fmax])

    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        array = mark.get_offsets()
        maskx = np.logical_and(array[:,0] >= min(x1,x2),  array[:,0] <= max(x1,x2))
        masky = np.logical_and(array[:,1] >= min(y1,y2),  array[:,1] <= max(y1,y2))
        mask = np.logical_and(maskx, masky)
        mark.set_offsets(array[np.logical_not(mask),:])
        fig.canvas.draw()
        print(f"select over {x1},{y1},{x2},{y2}")

    selector = RectangleSelector(ax, line_select_callback, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels',props={'edgecolor':'white', 'fill': False})
    selector.active = False
   
    ax.xaxis_date()
    date_format = mdates.DateFormatter("%F\n%H:%M:%S")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"Frequency (MHz) - {fcen * 1e-6:g} MHz")
    
    def add_point(scatter, point):
        array = scatter.get_offsets()
        array = np.vstack([array, point])
        scatter.set_offsets(array)
        fig.canvas.draw()

    def handle(key, x, y):
        print(f"pressed {key} over x={x} y={y}")
        if key == "d":
            selector.active = True
        elif key == "s":
            point = (x, y)
            add_point(line_fitting, point)
        elif key == "f":
            print("performing fitting on")
            print(line_fitting.get_offsets())

        sys.stdout.flush()

            
    def on_press(event):
        handle(event.key, event.xdata, event.ydata)
        sys.stdout.flush()

    def on_click(event):
        if event.button is MouseButton.MIDDLE:
            point = (event.xdata, event.ydata)
            add_point(mark, point)
            print(f"{event.xdata} {fcen + event.ydata}")
            sys.stdout.flush()

    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
