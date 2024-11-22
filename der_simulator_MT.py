#!/usr/bin/env python

###############################################
# IMPORT MODULES
###############################################

from timeit import default_timer as timer

import os
import sys
import numpy as np
from numpy.linalg import norm as norm
from npy_append_array import NpyAppendArray
from numba import njit

from functions_bd_mt import run_bd_mt



###############################################
# BD RUN ARGUMENTS
###############################################

chain = sys.argv[1]
if chain == "-h":
    print('\n# Usage: ./der_simulator_MT.py chain nuc_state n_sim nt nt_skip Nt alpha restart_flag\n')
    sys.exit()
chain = int(sys.argv[1])       # simulation index
nuc_state = sys.argv[2]        # nucleotide state (gtp or gdp)
n_sim =int(sys.argv[3])        # number of restarts in a chain of simulations
nt = int(sys.argv[4])          # number of steps in a single simulation
nt_skip = int(sys.argv[5])     # save trajectory every nt_skip steps
Nt = int(sys.argv[6])          # number of monomers in a PF
alpha = float(sys.argv[7])     # scaling factor for lateral energies
if len(sys.argv) == 9:
    restart_flag = sys.argv[8] # restart flag
else:
    restart_flag = ''



###############################################
# SET UP PARAMETER SPACE
###############################################

npf = 14

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
    params_stiff = np.array([
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
                              28.729752,             #20 epsilon_lat_homo, kJ/mol
                              21.021939,             #21 epsilon_lat_seam, kJ/mol
                              2.061726,              #22 a_lat_homo, shape, 1/nm
                              1.528234,              #23 a_lat_seam, shape, 1/nm
                              alpha                  #24 scaling factor for lateral energies
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
    params_stiff = np.array([
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
                              42.853914,             #20 epsilon_lat_homo, kJ/mol
                              25.857480,             #21 epsilon_lat_seam, kJ/mol
                              2.164717,              #22 a_lat_homo, shape, 1/nm
                              2.363691,              #23 a_lat_seam, shape, 1/nm
                              alpha                  #24 scaling factor for lateral energies
                            ])
else:
    print('\nNucleotide state can be either GTP or GDP!\n')
    sys.exit()



###############################################
# RUN BD AND WRITE FILES
###############################################

folder_save = 'sim_mt_%s_%d_%.2f_%s'  % (nuc_state, Nt, alpha, chain)

for i in range(n_sim):
    if restart_flag == '' and not os.path.exists('%s/traj_vert.npy'  % folder_save) and \
                              not os.path.exists('%s/traj_dir.npy'   % folder_save) and \
                              not os.path.exists('%s/traj_theta.npy' % folder_save) and \
                              not os.path.exists('%s/traj_U.npy'     % folder_save) and \
                              not os.path.exists('%s/traj_V.npy'     % folder_save) and \
                              not os.path.exists('%s/traj_mref.npy'  % folder_save):
        print("\nStarting a new BD simulation...")

        v_restart     = np.zeros((npf, Nt+1, 3))
        theta_restart = np.zeros((npf, Nt))
        ut_restart    = np.zeros((npf, Nt+1, 3))
        vt_restart    = np.zeros((npf, Nt+1, 3))
        mref_restart  = np.zeros((npf, Nt))

        # save files in a separate folder
        os.system('mkdir %s' % folder_save)
    elif restart_flag == '-r' and os.path.exists('%s/traj_vert.npy'  % folder_save) and \
                                  os.path.exists('%s/traj_dir.npy'   % folder_save) and \
                                  os.path.exists('%s/traj_theta.npy' % folder_save) and \
                                  os.path.exists('%s/traj_U.npy'     % folder_save) and \
                                  os.path.exists('%s/traj_V.npy'     % folder_save) and \
                                  os.path.exists('%s/traj_mref.npy'  % folder_save):
        print("\nRestarting from a previous BD simulation...")

        v_restart     = np.load('%s/traj_vert.npy'  % folder_save)[-1]
        theta_restart = np.load('%s/traj_theta.npy' % folder_save)[-1]
        ut_restart    = np.load('%s/traj_U.npy'     % folder_save)[-1]
        vt_restart    = np.load('%s/traj_V.npy'     % folder_save)[-1]
        mref_restart  = np.load('%s/traj_mref.npy'  % folder_save)[-1]
    else:
        print("\nUse -r flag to continue or remove files from the previous simulation!")
        print("Or there is nothing to restart from!\n")
        sys.exit()

    t_start = timer()
    traj_vert, traj_dir, traj_theta, traj_U, traj_V, traj_mref = run_bd_mt(nt, nt_skip, Nt, npf,
                                                                           restart_flag, v_restart, theta_restart, mref_restart, ut_restart, vt_restart,
                                                                           params_diff, params_means, params_stiff)
    t_end = timer()
    print('Total run time of cycle %d is %f sec\n' % (i, t_end - t_start))

    with NpyAppendArray('%s/traj_vert.npy'  % folder_save) as file_vert:
        file_vert.append(traj_vert)
    with NpyAppendArray('%s/traj_dir.npy'   % folder_save) as file_dir:
        file_dir.append(traj_dir)
    with NpyAppendArray('%s/traj_theta.npy' % folder_save) as file_theta:
        file_theta.append(traj_theta)
    with NpyAppendArray('%s/traj_U.npy'     % folder_save) as file_U:
        file_U.append(traj_U)
    with NpyAppendArray('%s/traj_V.npy'     % folder_save) as file_V:
        file_V.append(traj_V)
    with NpyAppendArray('%s/traj_mref.npy'  % folder_save) as file_mref:
        file_mref.append(traj_mref)

    if restart_flag == '':
        restart_flag = '-r'

