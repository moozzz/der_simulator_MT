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
from params_bd_run import *



###############################################
# SET UP PARAMETER SPACE
###############################################

# number of PFs and max PF length
Nt_array = np.array(Nt_array)
Nt_max = int(np.max(Nt_array))
npf = len(Nt_array)

# implicitly setting dt = 10.0 ps
params_diff = np.array([
                        2e-5*10.0, # nm^2, translational msd per time step
                        2e-5*10.0  # rad^2, rotational msd per time step
                       ])

if nuc_state == 'gtp':
    # NOTE: GTP
    # direct fit to distributions
    params_means = np.array([
                              4.165410341720552,     #0 beta ht, nm
                              3.7903647007818297,    #1 alpha ht, nm
                             -0.08888236830005103,   #2 intra K1eq
                             -0.30154522643205484,   #3 inter K1eq
                             -0.004020310251829198,  #4 intra K2eq
                              0.06212100038153915,   #5 inter K2eq
                             -0.01467755725478138,   #6 intra Mtwist_eq
                             -0.06508463306367768    #7 inter Mtwist_eq
                            ])

    # NOTE: GTP
    # optimized with fuzzy pso
    params_ener = np.array([
                              6868.738719509609,     #8  beta  Es, kJ/mol/nm
                              6061.611683827584,     #9  alpha Es, kJ/mol/nm
                              4353.6596828386155,    #10 intra Ek1, kJ/mol*nm
                              6724.2672374575195,    #11 inter Ek1, kJ/mol*nm
                              16138.435157608803,    #12 intra Ek2, kJ/mol*nm
                              16170.794588806031,    #13 inter Ek2, kJ/mol*nm
                              8036.790422397843,     #14 intra Et, kJ/mol*nm
                              7841.670229500417,     #15 inter Et, kJ/mol*nm
                              8434.290705567179,     #16 intra Etb2, kJ/mol*nm
                              8276.75849097318,      #17 inter Etb2, kJ/mol*nm
                              24.0*2.5,              #18 epsilon_long, kJ/mol
                              3.650588936,           #19 a_long, shape, 1/nm
                              1.0,                   #20 mode_long, 0 = harmonic, 1 = morse
                              28.729752,             #21 epsilon_lat_homo, kJ/mol
                              21.021939,             #22 epsilon_lat_seam, kJ/mol
                              2.061726,              #23 a_lat_homo, shape, 1/nm
                              1.528234,              #24 a_lat_seam, shape, 1/nm
                              alpha                  #25 scaling factor for lateral energies
                            ])
elif nuc_state == 'gdp':
    # NOTE: GDP
    # direct fit to distributions
    params_means = np.array([
                              4.199384550631562,     #0 beta ht, nm
                              3.7848421331195783,    #1 alpha ht, nm
                             -0.045074817041818344,  #2 intra K1eq
                             -0.2545584576936117,    #3 inter K1eq
                             -0.012235549342411618,  #4 intra K2eq
                              0.07200631260397554,   #5 inter K2eq
                             -0.01034837399031294,   #6 intra Mtwist_eq
                             -0.13818616328638605    #7 inter Mtwist_eq
                            ])

    # NOTE: GDP
    # optimized with fuzzy pso
    params_ener = np.array([
                              11397.296081806451,    #8  beta  Es, kJ/mol/nm
                              10492.293957440254,    #9  alpha Es, kJ/mol/nm
                              3250.9012385119554,    #10 intra Ek1, kJ/mol*nm
                              13115.55507249061,     #11 inter Ek1, kJ/mol*nm
                              18479.656451977902,    #12 intra Ek2, kJ/mol*nm
                              17889.84426829865,     #13 inter Ek2, kJ/mol*nm
                              12931.46870870545,     #14 intra Et, kJ/mol*nm
                              12783.867349750595,    #15 inter Et, kJ/mol*nm
                              9665.777748438924,     #16 intra Etb2, kJ/mol*nm
                              9039.674533560048,     #17 inter Etb2, kJ/mol*nm
                              24.0*2.5,              #18 epsilon_long, kJ/mol
                              4.806408919,           #19 a_long, shape, 1/nm
                              1.0,                   #20 mode_long, 0 = harmonic, 1 = morse
                              42.853914,             #21 epsilon_lat_homo, kJ/mol
                              25.857480,             #22 epsilon_lat_seam, kJ/mol
                              2.164717,              #23 a_lat_homo, shape, 1/nm
                              2.363691,              #24 a_lat_seam, shape, 1/nm
                              alpha                  #25 scaling factor for lateral energies
                            ])
else:
    print('\nNucleotide state can be either GTP or GDP!\n')
    sys.exit()



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
    traj_vert, traj_theta, traj_U, traj_V, traj_mref, traj_dir = run_bd_mt(nt, nt_skip, Nt_array, npf,
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

