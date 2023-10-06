#!/usr/bin/env python3
import torch
import sys

import numpy as np
import time
import os
import platform

#windows
print(f"Current platform:{platform.system()}")
if platform.system()=="Windows":
    dll_dirs=os.environ["PATH"].split(';')
    for dir in dll_dirs:
        if any([ (kw in dir) for kw in ["bin","lib","x64"]]):
            os.add_dll_directory(dir)

    sys.path.append("./build/src/Debug/")



import gpu_library

size = 1000000
arr1 = np.linspace(1.0,100.0, size)
arr2 = np.linspace(1.0,100.0, size)

runs = 10
factor = 3.0

t0 = time.time()
for _ in range(runs):
    gpu_library.multiply_with_scalar(arr1, factor)
print("gpu time: {}".format(time.time()-t0))
t0 = time.time()
for _ in range(runs):
    arr2 = arr2 * factor
print("cpu time: {}".format(time.time()-t0))

print("results match: {}".format(np.allclose(arr1,arr2)))
