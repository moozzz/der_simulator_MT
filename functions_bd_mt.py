###############################################
# IMPORT MODULES
###############################################

import numpy as np
from numpy.linalg import norm as norm
from numba import njit

from functions_pf import get_angle
from functions_pf import parallel_transport
from functions_pf import computeEdges
from functions_pf import computeTangents
from functions_pf import computeBishopFrame
from functions_pf import computeMaterialFrame
from functions_pf import computeVoronoiLen
from functions_pf import computeCurvatureBinormals
from functions_pf import computeTwist

from functions_forces import Fstretch
from functions_forces import Fbend
from functions_forces import Ftwist
from functions_forces import Ftwist_theta
from functions_forces import FcoupleM_k2
from functions_forces import Fcouple_theta2
from functions_forces import Flat



###############################################
# USER-DEFINED SUPPLEMENTARY FUNCTIONS
###############################################

@njit(fastmath=True)
def fill_params(v1, v2, Nt):
    params = np.zeros(Nt+1)

    for i in range(1, Nt):
        if i % 2 != 0:
            params[i] = v1
        else:
            params[i] = v2

    return params

@njit(fastmath=True)
def unpack_params(params_diff, params_means, params_ener, Nt_max):
    # diffision coefficients
    dv2 = params_diff[0]              # nm^2
    sqrt_2_dv2 = np.sqrt(2.0 * dv2)   # nm
    dth2 = params_diff[1]             # rad^2
    sqrt_2_dth2 = np.sqrt(2.0 * dth2) # rad

    # mean values
    ht        = fill_params(params_means[0], params_means[1], Nt_max) # nm
    K1eq      = fill_params(params_means[2], params_means[3], Nt_max)
    K2eq      = fill_params(params_means[4], params_means[5], Nt_max)
    Mtwist_eq = fill_params(params_means[6], params_means[7], Nt_max)
    ht[0] = 8.15 / 2.0

    # stiffness coefficients
    Es   = fill_params(params_ener[0], params_ener[1], Nt_max) # kJ/mol/nm
    Ek1  = fill_params(params_ener[2], params_ener[3], Nt_max) # kJ/mol*nm
    Ek2  = fill_params(params_ener[4], params_ener[5], Nt_max) # kJ/mol*nm
    Et   = fill_params(params_ener[6], params_ener[7], Nt_max) # kJ/mol*nm
    Etb2 = fill_params(params_ener[8], params_ener[9], Nt_max) # kJ/mol*nm

    # longitudinal bond parameters
    epsilon_long = params_ener[10] # kJ/mol
    a_long = params_ener[11]       # 1/nm
    mode_long = params_ener[12]    # 0 = harmonic, 1 = morse

    # lateral bond parameters
    epsilon_lat_homo = params_ener[13] # kJ/mol
    epsilon_lat_seam = params_ener[14] # kJ/mol
    a_lat_homo = params_ener[15]       # 1/nm
    a_lat_seam = params_ener[16]       # 1/nm
    alpha_lat = params_ener[17]

    return (dv2, sqrt_2_dv2, dth2, sqrt_2_dth2,
            ht, K1eq, K2eq, Mtwist_eq,
            Es, Ek1, Ek2, Et, Etb2,
            epsilon_long, a_long, mode_long,
            epsilon_lat_homo, epsilon_lat_seam, a_lat_homo, a_lat_seam, alpha_lat)

@njit(fastmath=True)
def init_start_conf(flag_restart, Nt_array, Nt_max, npf, ht,
                    v_restart, theta_restart, ut_restart, vt_restart, mref_restart):
    # hard-coded MT parameters
    R_MT = 12.0 # nm

    # initialize arrays
    v      = np.zeros((npf, Nt_max+1, 3))
    theta  = np.zeros((npf, Nt_max))
    ut     = np.zeros((npf, Nt_max, 3))
    vt     = np.zeros((npf, Nt_max, 3))
    mref   = np.zeros((npf, Nt_max))
    ed     = np.zeros((npf, Nt_max, 3))
    tang   = np.zeros((npf, Nt_max, 3))
    Mtwist = np.zeros((npf, Nt_max+1))
    M1     = np.zeros((npf, Nt_max, 3))
    M2     = np.zeros((npf, Nt_max, 3))
    lv     = np.zeros((npf, Nt_max+1))
    kb     = np.zeros((npf, Nt_max+1, 3))

    u0     = np.zeros((npf, 3))
    t0     = np.array([0.0, 0.0, 1.0])
    
    if flag_restart:
        # start from last frame
        v = v_restart
        theta = theta_restart
        ut = ut_restart
        vt = vt_restart
        mref = mref_restart
    else:
        # locate PF positions on a circle
        init_pf_pos = np.zeros((npf, 2))
        for p in range(npf):
            init_pf_pos[p, 0] = R_MT * np.cos(2.0 * np.pi * p / 14.0)
            init_pf_pos[p, 1] = R_MT * np.sin(2.0 * np.pi * p / 14.0)
    
        # initialize node positions
        for p in range(npf):
            temp_pos = 0.0
            for i in range(Nt_array[p]+1):
                v[p, i] = np.array([init_pf_pos[p, 0], init_pf_pos[p, 1], temp_pos - 0.8845 * p])
                temp_pos = ht[i] + temp_pos
    
        # initialize MT geometry
        for p in range(npf):
            u_tem = -np.array([init_pf_pos[p, 0], init_pf_pos[p, 1], 0.0])
            u0[p] = u_tem / norm(u_tem)
    
            ed[p] = computeEdges(Nt_array[p], Nt_max, v[p])
            tang[p] = computeTangents(Nt_array[p], Nt_max, ed[p])

            ut[p], vt[p] = computeBishopFrame(Nt_array[p], Nt_max, t0, u0[p], tang[p])
            Mtwist[p] = computeTwist(Nt_array[p], Nt_max, theta[p], mref[p])
            M1[p], M2[p] = computeMaterialFrame(Nt_array[p], Nt_max, ut[p], vt[p], theta[p])

            lv[p] = computeVoronoiLen(Nt_array[p], Nt_max, ed[p])
            kb[p] = computeCurvatureBinormals(Nt_array[p], Nt_max, tang[p])

            M1[p, 0] = u0[p]
            M2[p, 0] = np.cross(t0, u0[p])
            ut[p, 0] = u0[p]
            vt[p, 0] = np.cross(t0, u0[p])

    return (v, theta, ut, vt, mref, ed,
            tang, Mtwist, M1, M2, lv, kb)



###############################################
# BROWNIAN DYNAMICS MODULE
###############################################
@njit(fastmath=True)
def run_bd_mt(nt, nt_skip, Nt_array, npf, Nt_max, Nt_frozen, kbt, flag_restart, v_restart, theta_restart, mref_restart, ut_restart, vt_restart, params_diff, params_means, params_ener):

    np.random.seed(111)

    ################################
    # Unpacking model parameters
    ################################
    (dv2, sqrt_2_dv2, dth2, sqrt_2_dth2,
     ht, K1eq, K2eq, Mtwist_eq,
     Es, Ek1, Ek2, Et, Etb2,
     epsilon_long, a_long, mode_long,
     epsilon_lat_homo, epsilon_lat_seam, a_lat_homo, a_lat_seam, alpha_lat) = unpack_params(params_diff, params_means, params_ener, Nt_max)

    ################################
    # Starting configuration
    ################################
    (v, theta, ut, vt, mref, ed,
     tang, Mtwist, M1, M2, lv, kb) = init_start_conf(flag_restart, Nt_array, Nt_max, npf, ht,
                                                     v_restart, theta_restart, ut_restart, vt_restart, mref_restart)

    ################################
    # Output arrays
    ################################
    traj_v     = np.zeros((int(nt // nt_skip), npf, Nt_max+1, 3))
    traj_dir   = np.zeros((int(nt // nt_skip), npf, Nt_max, 3))
    traj_theta = np.zeros((int(nt // nt_skip), npf, Nt_max))
    traj_U     = np.zeros((int(nt // nt_skip), npf, Nt_max, 3))
    traj_V     = np.zeros((int(nt // nt_skip), npf, Nt_max, 3))
    traj_mref  = np.zeros((int(nt // nt_skip), npf, Nt_max))
    ut_1       = np.zeros((npf, Nt_max, 3))
    vt_1       = np.zeros((npf, Nt_max, 3))
    tang_1     = np.zeros((npf, Nt_max, 3))

    ################################
    # Main time cycle
    ################################
    frame = 0
    for ts in range(nt):
        ################################
        # Update PF
        ################################
        for p in range(npf):
            ed[p] = computeEdges(Nt_array[p], Nt_max, v[p])
            tang[p] = computeTangents(Nt_array[p], Nt_max, ed[p])

            if ts != 0:
                # parallel transport in time
                for i in range(Nt_array[p]):
                    ut[p, i] = parallel_transport(ut_1[p, i], tang_1[p, i], tang[p, i])
                    vt[p, i] = parallel_transport(vt_1[p, i], tang_1[p, i], tang[p, i])
                
                # update reference twist
                for i in range(1, Nt_array[p]):
                    uu = parallel_transport(ut[p, i-1], tang[p, i-1], tang[p, i])
                    mref[p, i] = get_angle(uu, ut[p, i], tang[p, i])

            M1[p], M2[p] = computeMaterialFrame(Nt_array[p], Nt_max, ut[p], vt[p], theta[p])
            lv[p] = computeVoronoiLen(Nt_array[p], Nt_max, ed[p])
            kb[p] = computeCurvatureBinormals(Nt_array[p], Nt_max, tang[p])
            Mtwist[p] = computeTwist(Nt_array[p], Nt_max, theta[p], mref[p])

        ################################
        # Compute Forces
        ################################
        # don't calculate lateral forces for a single PF
        if npf > 1:
            flv = Flat(Nt_array, M1, M2, v, tang, npf, epsilon_lat_homo, epsilon_lat_seam, a_lat_homo, a_lat_seam, alpha_lat)
        else:
            flv = np.zeros((npf, Nt_max+1, 3))

        for p in range(npf):
            fs = Fstretch(Nt_array[p], Nt_max, ed[p], tang[p], ht, Es, mode_long, epsilon_long, a_long)
            fb = Fbend(Nt_array[p], Nt_max, M1[p], M2[p], kb[p], tang[p], ed[p], lv[p], K1eq, K2eq, Ek1, Ek2)
            ft = Ftwist(Nt_array[p], Nt_max, ed[p], Mtwist[p], kb[p], lv[p], Mtwist_eq, Et)
            ft_theta = Ftwist_theta(Nt_array[p], Nt_max, Mtwist[p], lv[p], Mtwist_eq, Et)
            fc2 = FcoupleM_k2(Nt_array[p], Nt_max, Mtwist[p], kb[p], lv[p], tang[p], ed[p], M1[p], Mtwist_eq, K2eq, Etb2)
            fc2_theta = Fcouple_theta2(Nt_array[p], Nt_max, M1[p], lv[p], kb[p], K2eq, Etb2)

            ################################
            # Update coordinates and angles
            ################################
            # update coordinates
            for i in range(Nt_frozen+1, Nt_array[p]+1):
                v[p, i] = v[p, i] +\
                          dv2 / kbt * (fs[i] + fb[i] + ft[i] + fc2[i] + flv[p, i]) +\
                          sqrt_2_dv2 * np.array([np.random.normal(),
                                                 np.random.normal(),
                                                 np.random.normal()])

            # update angles
            for i in range(Nt_frozen, Nt_array[p]):
                theta[p, i] = theta[p, i] +\
                              dth2 / kbt * (ft_theta[i] + fc2_theta[i]) +\
                              sqrt_2_dth2 * np.random.normal()

            ################################
            # Save for parallel transport
            ################################
            tang_1[p] = tang[p]
            ut_1[p] = ut[p]
            vt_1[p] = vt[p]

        ################################
        # Write trajectory
        ################################
        if ts % nt_skip == 0:
            for p in range(npf):
                for i in range(Nt_array[p]):
                    traj_dir[frame, p, i] = v[p, i] + (v[p, i+1] - v[p, i]) / 2.0 + M1[p, i] * 2.0
            
            traj_v[frame] = v
            traj_theta[frame] = theta
            traj_U[frame] = ut
            traj_V[frame] = vt
            traj_mref[frame] = mref
            
            frame += 1

    return traj_v, traj_theta, traj_U, traj_V, traj_mref, traj_dir

