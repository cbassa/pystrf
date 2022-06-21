#!/usr/bin/env python3
import sys
import argparse

import numpy as np
from strf.rfio import Spectrogram

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets  import RectangleSelector
import matplotlib as mpl

from modest import imshow

if __name__ == "__main__":
    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.fullscreen'].remove('f')
    mpl.rcParams['backend'] = "TkAgg"

    parser = argparse.ArgumentParser(description='rfplot: plot RF observations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', help='Input path to parent directory /a/b/')
    parser.add_argument('-P', help='Filename prefix c in c_??????.bin')
    parser.add_argument('-s', type=int, default=0,  help='Number of starting subintegration')
    parser.add_argument('-l', type=int, default=3600,  help='Number of subintegrations to plot')
    parser.add_argument('-C', type=int,  help='Site ID')
    
    args = parser.parse_args()
    # Read spectrogram
    s = Spectrogram(args.p, args.P, args.s, args.l, args.C)

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
    # imshow(ax, s.z,  vmin=vmin, vmax=vmax)
    imshow(ax, s.z, origin="lower", aspect="auto", interpolation="None",
              vmin=vmin, vmax=vmax,
              extent=[tmin, tmax, fmin, fmax])

    mode = {
        "current_mode" : None
    }
    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if mode["current_mode"] =="fit":
            t1_ind = round(len(s.t) * (x1 - tmin) / (tmax - tmin))
            t2_ind = round(len(s.t) * (x2 - tmin) / (tmax - tmin))
            f1_ind = round(len(s.freq) * (y1 - fmin) / (fmax - fmin))
            f2_ind = round(len(s.freq) * (y2 - fmin) / (fmax - fmin))
            
            submat = s.z[f1_ind:f2_ind,t1_ind:t2_ind]
            # TODO perform some action on submat
        elif mode["current_mode"] == "delete":
            array = mark.get_offsets()
            maskx = np.logical_and(array[:,0] >= min(x1,x2),  array[:,0] <= max(x1,x2))
            masky = np.logical_and(array[:,1] >= min(y1,y2),  array[:,1] <= max(y1,y2))
            mask = np.logical_and(maskx, masky)
            mark.set_offsets(array[np.logical_not(mask),:])
        fig.canvas.draw()
        current_mode = mode["current_mode"]
        print(f"select over {x1},{y1},{x2},{y2} in {current_mode} mode")

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
        print(array)
        array = np.vstack([array, point])
        scatter.set_offsets(array)
        fig.canvas.draw()

    def handle(key, x, y):
        print(f"pressed {key} over x={x} y={y}")
        if key == "d":
            selector.active = True
            mode["current_mode"] = "delete"
        elif key == "s":
            point = (x, y)
            add_point(line_fitting, point)
        elif key == "f":
            print("performing fitting on")
            mode["current_mode"] = "fit"
            selector.active = True
            print(line_fitting.get_offsets())
        elif key == "r":
            print("performing reset")
            mode["current_mode"] = None
            selector.active = False
            mark.set_offsets(np.empty((0, 2), float))
            line_fitting.set_offsets(np.empty((0, 2), float))
            fig.canvas.draw()

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
