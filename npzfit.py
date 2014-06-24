"""
Find the relationship between the Kreuz measure and Sin (no noise) using data
from npz files (given a directory).
"""
from scipy.optimize import curve_fit
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

# ignore cases with sigma > 0
npss = np.array(npss)
kreuz = np.array(kreuz)
sigma = np.array(sigma)
s0 = sigma < 0.0001
npss = npss[s0]
kreuz = kreuz[s0]

pnpss, _ = curve_fit(fit_func, npss, kreuz)
pkreuz, _ = curve_fit(fit_func, kreuz, npss)
print("Curve fit complete:")
print("M = %f D^3 + %f D^2 + %f D + %f" % tuple(pnpss))
print("D = %f M^3 + %f M^2 + %f M + %f" % tuple(pkreuz))
print("Plotting ...")
plt.figure(1)
plt.scatter(npss, kreuz)
plt.plot(np.sort(npss), fit_func(np.sort(npss), *pnpss), 'r--', linewidth=2)
plt.xlabel("$M$")
plt.ylabel("$D_s$")
plt.figure(2)
plt.scatter(npss, fit_func(kreuz, *pkreuz))
plt.plot([0, 1], 'r--', linewidth=2)
plt.xlabel("$M$")
plt.ylabel("$f(D_s)$")
plt.figure(3)
plt.scatter(kreuz, fit_func(npss, *pnpss))
plt.plot([0, max(kreuz)], [0, max(kreuz)], 'r--', linewidth=2)
plt.xlabel("$D_s$")
plt.ylabel("$g(M)$")
plt.show()
print("DONE!")
