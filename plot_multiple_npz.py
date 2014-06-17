"""
Collect all the data from the npz files generated using npss_kreuz_mc.py and
plot the data in many wonderful ways (or just as a scatter plot).
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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
for idx, npz in enumerate(npzfiles):
    npzdata = np.load(npz)
    npss.extend(npzdata["npss"])
    kreuz.extend(npzdata["kreuz"])
    sync.extend(npzdata["S_in"])
    if "sigma_in" in npzdata.files:
        sigma.extend(npzdata["sigma_in"])
    else:
        ndata = len(npzdata["npss"])
        sigma.extend([0]*ndata)

npss = np.array(npss)
kreuz = np.array(kreuz)
npss_kr = np.sqrt(1-kreuz/0.3)-0.0148
errors = npss-npss_kr
plt.scatter(npss, npss_kr)
plt.plot([0, 1], [0, 1], "k--")
#for n, e in zip(npss, errors):
#    plt.plot([n, n], [n, n-e], "b-")
plt.xlabel("NPSS")
plt.ylabel("Spike distance (rescaled)")
plt.show()
