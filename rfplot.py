#!/usr/bin/env python3
import sys
import argparse
import os 

import numpy as np
from strf.rfio import Spectrogram, get_site_info, get_frequency_info, get_satellite_info

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets  import RectangleSelector
import matplotlib as mpl
from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84, utc

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
    parser.add_argument('-C', type=int,  help='Site ID', default=4171)
    parser.add_argument('-F', help='List with frequencies')
    
    args = parser.parse_args()

    if "ST_DATADIR" not in os.environ:
        print("ST_DATADIR variable not found")
        sys.exit(1)

    site_fname = os.path.join(os.environ["ST_DATADIR"], "data", "sites.txt") 
    if not os.path.exists(site_fname):
        print(f"Sites file not available under {site_fname}")
        sys.exit(1)

    site = get_site_info(site_fname, args.C)
    if site is None:
        print(f"Site with no: {args.C} does not exist")
        sys.exit(1)

    site_location = wgs84.latlon(site["lat"], site["lon"], site["height"])
    
    if args.F is not None:
        freq_fname = args.F
    else:
        freq_fname = os.path.join(os.environ["ST_DATADIR"], "data", "frequencies.txt") 
    
    if "ST_TLEDIR" not in os.environ:
        print("ST_TLEDIR variable not found")
        sys.exit(1)

    tle_fname = os.path.join(os.environ["ST_TLEDIR"], "bulk.tle")
    
    # Read spectrogram
    s = Spectrogram(args.p, args.P, args.s, args.l, args.C)

    # Create plot
    vmin, vmax = np.percentile(s.z, (5, 99.95))

    # Time limits
    tmin, tmax = mdates.date2num(s.t[0]), mdates.date2num(s.t[-1])

    # Frequency limits
    fcen = np.mean(s.freq)
    fmin, fmax = (s.freq[0] - fcen) * 1e-6, (s.freq[-1] - fcen) * 1e-6
    ts = load.timescale()
    frequencies = []
    satellite_info = []

    if not os.path.exists(freq_fname):
        print(f"warning: Frequencies file not available under {freq_fname}")
    else:
        frequencies = get_frequency_info(freq_fname, fcen, s.freq[0], s.freq[-1])
        if not os.path.exists(tle_fname):
            print(f"TLE data not available under {tle_fname}")
            sys.exit(1)
        
        names = ('rise', 'culminate', 'set')
        t0,t1 = ts.utc(s.t[0].replace(tzinfo=utc)), ts.utc(s.t[-1].replace(tzinfo=utc))
        satellite_info = get_satellite_info(tle_fname, frequencies)

    print(f"Found {len(frequencies)} matching satellites")

    fig, ax = plt.subplots(figsize=(10, 6)) 
    mark = ax.scatter([], [],c="white",s=5)
    line_fitting = ax.scatter([], [], edgecolors="yellow",s=10, facecolors='none')
    # imshow(ax, s.z,  vmin=vmin, vmax=vmax)
    timestamps = [ x.replace(tzinfo=utc) for x in  s.t]
    for sat_info in satellite_info:
        satellite = EarthSatellite(sat_info["tle"][-2], sat_info["tle"][-1])
        t, events = satellite.find_events(site_location, t0, t1, altitude_degrees=0.0)
        if len(t) > 0:
            pairs = [ (ti, event)  for ti, event in zip(t, events)]
            if pairs[0][1] in [1,2]:
                pairs = [ (t0, 0)  ] + pairs # pad with rise
            
            if pairs[-1][1] in [0, 1]:
                pairs =  pairs + [ (t1, 2)  ] # pad with set

            pairs = [ (ti, event)  for ti, event in pairs if event != 1 ] # remove culminations

            sat_info["timeslot"] = [ (pairs[i][0].utc_datetime(), pairs[i+1][0].utc_datetime()) for i in range(0, len(pairs), 2)]

            for timeslot in sat_info["timeslot"]:
                selected_timestamps = [ x for x in timestamps if x >= timeslot[0] and x <= timeslot[1]]
                pos = (satellite - site_location).at(ts.utc(selected_timestamps))
                _, _, _, _, _, range_rate = pos.frame_latlon_and_rates(site_location)
                C = 299792.458 # km/s

                for freq in sat_info["frequencies"]:
                    freq1 =  (freq -  fcen * 1e-6)
                    dfreq = freq1 - range_rate.km_per_s / C * freq # MHz
                    ax.plot([mdates.date2num(x) for x in selected_timestamps], dfreq,c="lime")

    image = imshow(ax, s.z, origin="lower", aspect="auto", interpolation="None",
              vmin=vmin, vmax=vmax,
              extent=[tmin, tmax, fmin, fmax])

    mode = {
        "current_mode" : None,
        "vmin" : vmin,
        "vmax" : vmax
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
        elif key == "b":
            mode["vmax"] *= 0.95
            image.set_clim(mode["vmin"],mode["vmax"])
            fig.canvas.draw()
        elif key == "v":
            mode["vmax"] *= 1.05
            image.set_clim(mode["vmin"],mode["vmax"])
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
