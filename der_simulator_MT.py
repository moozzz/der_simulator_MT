#!/usr/bin/env python

###############################################
# IMPORT MODULES
###############################################

import os
import sys
import numpy as np
from npy_append_array import NpyAppendArray
from timeit import default_timer as timer

from functions_bd_mt import run_bd_mt
from params_bd_ff import *



###############################################
# RUN BD AND WRITE FILES
###############################################

folder_save = 'sim_mt_%s_%d_%.8f_%d'  % (nuc_state, Nt_max, alpha, chain)
folder_files = ['traj_vert.npy', 'traj_theta.npy', 'traj_U.npy', 'traj_V.npy', 'traj_mref.npy', 'traj_dir.npy']

for i in range(n_sim):
    # initialize new or restart old simulation
    if restart_flag == '' and not any(os.path.exists('%s/%s' % (folder_save, f)) for f in folder_files):
        print("\nStarting a new BD simulation...")

        v_restart     = np.zeros((npf, Nt_max+1, 3))
        theta_restart = np.zeros((npf, Nt_max))
        ut_restart    = np.zeros((npf, Nt_max+1, 3))
        vt_restart    = np.zeros((npf, Nt_max+1, 3))
        mref_restart  = np.zeros((npf, Nt_max))

        os.makedirs(folder_save)
    elif restart_flag == '-r' and all(os.path.exists('%s/%s' % (folder_save, f)) for f in folder_files):
        print("\nRestarting from a previous BD simulation...")

        v_restart     = np.load('%s/%s' % (folder_save, folder_files[0]))[-1]
        theta_restart = np.load('%s/%s' % (folder_save, folder_files[1]))[-1]
        ut_restart    = np.load('%s/%s' % (folder_save, folder_files[2]))[-1]
        vt_restart    = np.load('%s/%s' % (folder_save, folder_files[3]))[-1]
        mref_restart  = np.load('%s/%s' % (folder_save, folder_files[4]))[-1]
    else:
        print("\nUse -r flag to continue or remove files from the previous simulation!")
        print("Or there is nothing to restart from!\n")
        sys.exit()

    # run BD
    t_start = timer()
    traj_vert, traj_theta, traj_U, traj_V, traj_mref, traj_dir = run_bd_mt(nt, nt_skip, Nt_array, npf, Nt_max, Nt_frozen,
                                                                           restart_flag, v_restart, theta_restart, mref_restart, ut_restart, vt_restart,
                                                                           params_diff, params_means, params_ener)
    t_end = timer()

    # estimate performance
    print('Total run time of cycle %d is %f sec' % (i, t_end - t_start))
    if i == 0:
        print('Performance will be estimated in the next cycle')
    else:
        print('Performance in cycle %d is %.1f µs/day' % (i, (nt*10.0/1e6) / ((t_end - t_start)/60.0/60.0/24.0) ))

    # append trajectory data to the respective files
    for traj_x, f in zip([traj_vert, traj_theta, traj_U, traj_V, traj_mref, traj_dir], folder_files):
        with NpyAppendArray('%s/%s' % (folder_save, f)) as file:
            file.append(traj_x)

    if restart_flag == '':
        restart_flag = '-r'

