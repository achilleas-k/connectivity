import os
import sys
import numpy as np
import glob

directory = "."
if len(sys.argv) > 1:
    directory = sys.argv[1]
npzglob = os.path.join(directory, "*.npz")
npzfiles = glob.glob(npzglob)

nfiles = len(npzfiles)
flaggedfiles = []
for idx, npz in enumerate(npzfiles):
    npzdata = np.load(npz)
    if "sigma_in" in npzdata.files:
        sigma = np.array(npzdata["sigma_in"])
        npss = np.array(npzdata["npss"])
        if np.count_nonzero(sigma) > 0 and np.count_nonzero(npss) == 0:
            flaggedfiles.append(npz)
