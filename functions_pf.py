###############################################
# IMPORT MODULES
###############################################

import numpy as np
from numba import njit



###############################################
# USER-DEFINED GEOMETRIC FUNCTIONS
###############################################

@njit(fastmath=True)
def get_angle(vec1, vec2, vn):
    cross_v12_x = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    cross_v12_y = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    cross_v12_z = vec1[0]*vec2[1] - vec1[1]*vec2[0]

    return np.arctan2(cross_v12_x*vn[0] + cross_v12_y*vn[1] + cross_v12_z*vn[2],
                      vec1[0]*vec2[0]   + vec1[1]*vec2[1]   + vec1[2]*vec2[2])

@njit(fastmath=True)
def parallel_transport(u, t1, t2):
    b0 = t1[1]*t2[2] - t1[2]*t2[1]
    b1 = t1[2]*t2[0] - t1[0]*t2[2]
    b2 = t1[0]*t2[1] - t1[1]*t2[0]
    b_norm = np.sqrt(b0*b0 + b1*b1 + b2*b2)

    if b_norm == 0.0:
        return u

    inv_b_norm = 1.0 / b_norm
    b0 *= inv_b_norm
    b1 *= inv_b_norm
    b2 *= inv_b_norm

    n1_0 = t1[1]*b2 - t1[2]*b1
    n1_1 = t1[2]*b0 - t1[0]*b2
    n1_2 = t1[0]*b1 - t1[1]*b0

    n2_0 = t2[1]*b2 - t2[2]*b1
    n2_1 = t2[2]*b0 - t2[0]*b2
    n2_2 = t2[0]*b1 - t2[1]*b0

    d1 = u[0]*t1[0] + u[1]*t1[1] + u[2]*t1[2]
    d2 = u[0]*n1_0  + u[1]*n1_1  + u[2]*n1_2
    d3 = u[0]*b0    + u[1]*b1    + u[2]*b2

    return np.array([d1*t2[0] + d2*n2_0 + d3*b0,
                     d1*t2[1] + d2*n2_1 + d3*b1,
                     d1*t2[2] + d2*n2_2 + d3*b2])

@njit(fastmath=True)
def computeEdges(Nt, Nt_max, v):
    ed = np.zeros((Nt_max, 3))

    ed[:Nt] = v[1:Nt + 1] - v[:Nt]

    return ed

@njit(fastmath=True)
def computeEdgeNorms(Nt, Nt_max, ed):
    ed_norms = np.zeros(Nt_max)

    for i in range(Nt):
        ed_norms[i] = np.sqrt(ed[i, 0]*ed[i, 0] + ed[i, 1]*ed[i, 1] + ed[i, 2]*ed[i, 2])

    return ed_norms

@njit(fastmath=True)
def computeTangents(Nt, Nt_max, ed, ed_norms):
    tang = np.zeros((Nt_max, 3))

    for i in range(Nt):
        inv_ed_norms = 1.0 / ed_norms[i]
        tang[i, 0] = ed[i, 0] * inv_ed_norms
        tang[i, 1] = ed[i, 1] * inv_ed_norms
        tang[i, 2] = ed[i, 2] * inv_ed_norms

    return tang

@njit(fastmath=True)
def computeBishopFrame(Nt, Nt_max, t0, u0, tang):
    u = u0
    U = np.zeros((Nt_max, 3))
    V = np.zeros((Nt_max, 3))

    for i in range(Nt):
        t1 = tang[i]
        u = parallel_transport(u, t0, t1)
        inv_u = 1.0 / np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
        u = u * inv_u
        U[i] = u
        V[i, 0] = t1[1]*u[2] - t1[2]*u[1]
        V[i, 1] = t1[2]*u[0] - t1[0]*u[2]
        V[i, 2] = t1[0]*u[1] - t1[1]*u[0]
        t0 = t1

    return U, V

@njit(fastmath=True)
def computeMaterialFrame(Nt, Nt_max, U, V, theta):
    M1 = np.zeros((Nt_max, 3))
    M2 = np.zeros((Nt_max, 3))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    M1 =  cos_theta[:, None] * U + sin_theta[:, None] * V
    M2 = -sin_theta[:, None] * U + cos_theta[:, None] * V

    return M1, M2

@njit(fastmath=True)
def computeVoronoiLen(Nt, Nt_max, ed_norms):
    lv = np.zeros(Nt_max+1)

    lv[0]  = 0.5 * ed_norms[0]
    lv[Nt] = 0.5 * ed_norms[Nt-1]
    for i in range(1, Nt):
        lv[i] = 0.5 * (ed_norms[i-1] + ed_norms[i])

    return lv

@njit(fastmath=True)
def computeCurvatureBinormals(Nt, Nt_max, tang):
    kb = np.zeros((Nt_max+1, 3))

    for i in range(1, Nt):
        inv_denom = 2.0 / (1.0 + tang[i-1, 0]*tang[i, 0] + tang[i-1, 1]*tang[i, 1] + tang[i-1, 2]*tang[i, 2])
        kb[i, 0] = (tang[i-1, 1]*tang[i, 2] - tang[i-1, 2]*tang[i, 1]) * inv_denom
        kb[i, 1] = (tang[i-1, 2]*tang[i, 0] - tang[i-1, 0]*tang[i, 2]) * inv_denom
        kb[i, 2] = (tang[i-1, 0]*tang[i, 1] - tang[i-1, 1]*tang[i, 0]) * inv_denom

    return kb

@njit(fastmath=True)
def computeTwist(Nt, Nt_max, theta, mref):
    Mtwist = np.zeros(Nt_max+1)

    for i in range(1, Nt):
        Mtwist[i] = theta[i] - theta[i-1] + mref[i]

    return Mtwist

@njit(fastmath=True)
def computeK(Nt, Nt_max, M, kb, sign):
    K = np.zeros(Nt_max+1)

    for i in range(1, Nt):
        K[i] = sign * 0.5 * ((M[i-1, 0] + M[i, 0])*kb[i, 0] +\
                             (M[i-1, 1] + M[i, 1])*kb[i, 1] +\
                             (M[i-1, 2] + M[i, 2])*kb[i, 2])

    return K



###########################################################
# USER-DEFINED PARTIAL DERIVATIVES
###########################################################

@njit(fastmath=True)
def computedMde(Nt, Nt_max, ed_norms, kb, sameIndex):
    dMde = np.zeros((Nt_max+1, 3))

    for i in range(1, Nt):
        if sameIndex:
            dMde[i] = kb[i] / (2.0 * ed_norms[i])
        else:
            dMde[i] = kb[i] / (2.0 * ed_norms[i-1])

    return dMde

@njit(fastmath=True)
def computedUdM(Nt, Nt_max, Mtwist, lv, Mtwist_eq):
    dUdM = np.zeros(Nt_max+1)

    for i in range(1, Nt):
        dUdM[i] = 1.0 / lv[i] * (Mtwist[i] - Mtwist_eq[i])

    return dUdM

@njit(fastmath=True)
def computedKde(Nt, Nt_max, M1, M2, K1, K2, kb, tang, ed_norms):
    dK1de_same = np.zeros((Nt_max+1, 3))
    dK1de_diff = np.zeros((Nt_max+1, 3))
    dK2de_same = np.zeros((Nt_max+1, 3))
    dK2de_diff = np.zeros((Nt_max+1, 3))

    for i in range(1, Nt):
        inv_denom = 1.0 / (1.0 + tang[i-1, 0]*tang[i, 0] + tang[i-1, 1]*tang[i, 1] + tang[i-1, 2]*tang[i, 2])
        inv_ei    = 1.0 / ed_norms[i]
        inv_eprev = 1.0 / ed_norms[i-1]

        Ttilda_0  = (tang[i-1, 0] + tang[i, 0]) * inv_denom
        Ttilda_1  = (tang[i-1, 1] + tang[i, 1]) * inv_denom
        Ttilda_2  = (tang[i-1, 2] + tang[i, 2]) * inv_denom

        Mtilda_M2_0 = (M2[i-1, 0] + M2[i, 0]) * inv_denom
        Mtilda_M2_1 = (M2[i-1, 1] + M2[i, 1]) * inv_denom
        Mtilda_M2_2 = (M2[i-1, 2] + M2[i, 2]) * inv_denom

        Mtilda_M1_0 = (M1[i-1, 0] + M1[i, 0]) * inv_denom
        Mtilda_M1_1 = (M1[i-1, 1] + M1[i, 1]) * inv_denom
        Mtilda_M1_2 = (M1[i-1, 2] + M1[i, 2]) * inv_denom

        cross_tprev_M2_0 = tang[i-1, 1]*Mtilda_M2_2 - tang[i-1, 2]*Mtilda_M2_1
        cross_tprev_M2_1 = tang[i-1, 2]*Mtilda_M2_0 - tang[i-1, 0]*Mtilda_M2_2
        cross_tprev_M2_2 = tang[i-1, 0]*Mtilda_M2_1 - tang[i-1, 1]*Mtilda_M2_0

        cross_ti_M2_0 = tang[i, 1]*Mtilda_M2_2 - tang[i, 2]*Mtilda_M2_1
        cross_ti_M2_1 = tang[i, 2]*Mtilda_M2_0 - tang[i, 0]*Mtilda_M2_2
        cross_ti_M2_2 = tang[i, 0]*Mtilda_M2_1 - tang[i, 1]*Mtilda_M2_0

        cross_tprev_M1_0 = tang[i-1, 1]*Mtilda_M1_2 - tang[i-1, 2]*Mtilda_M1_1
        cross_tprev_M1_1 = tang[i-1, 2]*Mtilda_M1_0 - tang[i-1, 0]*Mtilda_M1_2
        cross_tprev_M1_2 = tang[i-1, 0]*Mtilda_M1_1 - tang[i-1, 1]*Mtilda_M1_0

        cross_ti_M1_0 = tang[i, 1]*Mtilda_M1_2 - tang[i, 2]*Mtilda_M1_1
        cross_ti_M1_1 = tang[i, 2]*Mtilda_M1_0 - tang[i, 0]*Mtilda_M1_2
        cross_ti_M1_2 = tang[i, 0]*Mtilda_M1_1 - tang[i, 1]*Mtilda_M1_0

        dK1de_same[i, 0] = inv_ei    * (-K1[i] * Ttilda_0 - cross_tprev_M2_0)
        dK1de_same[i, 1] = inv_ei    * (-K1[i] * Ttilda_1 - cross_tprev_M2_1)
        dK1de_same[i, 2] = inv_ei    * (-K1[i] * Ttilda_2 - cross_tprev_M2_2)
        dK1de_diff[i, 0] = inv_eprev * (-K1[i] * Ttilda_0 + cross_ti_M2_0)
        dK1de_diff[i, 1] = inv_eprev * (-K1[i] * Ttilda_1 + cross_ti_M2_1)
        dK1de_diff[i, 2] = inv_eprev * (-K1[i] * Ttilda_2 + cross_ti_M2_2)
        dK2de_same[i, 0] = inv_ei    * (-K2[i] * Ttilda_0 + cross_tprev_M1_0)
        dK2de_same[i, 1] = inv_ei    * (-K2[i] * Ttilda_1 + cross_tprev_M1_1)
        dK2de_same[i, 2] = inv_ei    * (-K2[i] * Ttilda_2 + cross_tprev_M1_2)
        dK2de_diff[i, 0] = inv_eprev * (-K2[i] * Ttilda_0 - cross_ti_M1_0)
        dK2de_diff[i, 1] = inv_eprev * (-K2[i] * Ttilda_1 - cross_ti_M1_1)
        dK2de_diff[i, 2] = inv_eprev * (-K2[i] * Ttilda_2 - cross_ti_M1_2)

    return dK1de_same, dK1de_diff, dK2de_same, dK2de_diff

@njit(fastmath=True)
def computedUdK(Nt, Nt_max, K, lv, Keq):
    dUdK = np.zeros(Nt_max+1)

    for i in range(1, Nt):
        dUdK[i] = 1.0 / lv[i] * (K[i] - Keq[i])

    return dUdK

