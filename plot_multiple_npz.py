"""
Collect all the data from the npz files generated using npss_kreuz_mc.py and
plot the data in many wonderful ways (or just as a scatter plot).
"""

import sys
import os
import glob
import numpy as np

directory = "."
if len(sys.argv) > 2:
    directory = sys.argv[2]
npzglob = os.path.join(directory, "*.npz")
npzfiles = glob.glob(npzglob)

for npz in npzfiles:
    npzdata = 
