"""
Find the relationship between the Kreuz measure and Sin (no noise) using data
from npz files (given a directory).
"""
from scipy.optimize import curve_fit
import glob

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def fit_func(dist, a, b, c, d):
    return a*dist**3+b*dist**2+c*dist+d


directory = "."
if len(sys.argv) > 1:
    directory = sys.argv[1]
npzglob = os.path.join(directory, "*.npz")
npzfiles = glob.glob(npzglob)

nfiles = len(npzfiles)
npss = []
kreuz = []
sync = []
sigma = []
N_in = []
r_in = []
w_in = []

for idx, npz in enumerate(npzfiles):
    npzdata = np.load(npz)
    N_in.extend(npzdata["N_in"])
    r_in.extend(npzdata["r_in"])
    w_in.extend(npzdata["w_in"])
    npss.extend(npzdata["npss"])
    kreuz.extend(npzdata["kreuz"])
    sync.extend(npzdata["S_in"])
    if "sigma_in" in npzdata.files:
        sigma.extend(npzdata["sigma_in"])
    else:
        ndata = len(npzdata["npss"])
        sigma.extend([0]*ndata)


popt, pcov = curve_fit(fit_func, npss, kreuz)
print("Plotting ...")
plt.scatter(npss, kreuz)
plt.plot(npss, fit_func(npss, *popt), 'r--', linewidth=2)
plt.show()
print("DONE!")
