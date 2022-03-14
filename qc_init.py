import logging
import argparse
from hashlib import sha1
import os.path

import signac
import numpy as np
## Note: kT 0.38, sp_phi 0.65, epsilon_ratio 2.6 lambdasigma 0.1

## state points ##
n_edges = int(3)                          # n_edges
patch_offset = 1                                   # patch_offset
n_repeats = int(100)       # n_repeats
seed = 9487                            # replica
pressure = 1.5                      # pressure
use_floppy_box = True
kT_end = 0.1# np.round(np.linspace(0.02, 0.2, 10), 2).tolist()
kT_init = 1.0
epsilon_ratio = 2.6 # np.round(np.linspace(2.1, 3.1, 11), 2).tolist()
sp_phi = [0.62] # np.round(np.linspace(0.61, 0.69, 5), 2).tolist()
sigma = 1
lambdasigma = 0.1 # np.round(np.linspace(0.05, 0.15, 11), 2).tolist()
repulsive_radius = 0.54 # np.round(np.linspace(0.5, 0.59, 10), 2).tolist()

def main():
    project = signac.init_project('qc_large_system')
    #project = signac.get_project()
    for arg1 in kT:
        for arg2 in sp_phi:
            sp = {'n_edges': n_edges,
                  'patch_offset': patch_offset,
                  'n_repeats': n_repeats,
                  'seed': seed,
                  'kT_end': kT_end,
                  'kT_init': kT_init,
                  'pressure': pressure,
                  'use_floppy_box': use_floppy_box,
                  'epsilon_ratio': epsilon_ratio,
                  'sp_phi': arg2,
                  'sigma': sigma,
                  'lambdasigma': lambdasigma,
                  'repulsive_radius': repulsive_radius
            }
            job = project.open_job(sp).init()

if __name__ == '__main__':
    main()
