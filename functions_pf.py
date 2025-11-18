###############################################
# IMPORT MODULES
###############################################

import numpy as np
from numpy.linalg import norm as norm
from numba import njit



###############################################
# USER-DEFINED GEOMETRIC FUNCTIONS
###############################################

@njit(fastmath=True)
def get_angle(vec1, vec2, vn):
    return np.arctan2( np.dot(np.cross(vec1, vec2), vn), np.dot(vec1, vec2) )

@njit(fastmath=True)
def parallel_transport(u, t1, t2):
    b = np.cross(t1, t2)

    if norm(b) == 0.0:
        return u

    b = b / norm(b)
    n1 = np.cross(t1, b)
    n2 = np.cross(t2, b)

    return np.dot(u, t1)*t2 + np.dot(u, n1)*n2 + np.dot(u, b)*b

@njit(fastmath=True)
def computeEdges(Nt, Nt_max, v):
    ed = np.zeros((Nt_max, 3))

    ed[:Nt] = v[1:Nt + 1] - v[:Nt]

    return ed

@njit(fastmath=True)
def computeBishopFrame(Nt, Nt_max, t0, u0, tang):
    u = u0
    U = np.zeros((Nt_max, 3))
    V = np.zeros((Nt_max, 3))

    for i in range(Nt):
        t1 = tang[i]
        u = parallel_transport(u, t0, t1)
        u = u / norm(u)
        v = np.cross(t1, u)
        U[i] = u
        V[i] = v
        t0 = t1

    return U, V

@njit(fastmath=True)
def computeMaterialFrame(Nt, Nt_max, U, V, theta):
    M1 = np.zeros((Nt_max+1, 3))
    M2 = np.zeros((Nt_max+1, 3))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    M1 =  cos_theta[:, None] * U + sin_theta[:, None] * V
    M2 = -sin_theta[:, None] * U + cos_theta[:, None] * V

    return M1, M2

@njit(fastmath=True)
def computeVoronoiLen(Nt, Nt_max, ed):
    lv = np.zeros(Nt_max+1)

    ed_norms = np.array([norm(x) for x in ed])
    lv[0]  = 0.5 * ed_norms[0]
    lv[Nt] = 0.5 * ed_norms[Nt-1]

    for i in range(1, Nt):
        lv[i] = 0.5 * (ed_norms[i-1] + ed_norms[i])

    return lv

@njit(fastmath=True)
def computeCurvatureBinormals(Nt, Nt_max, tang):
    kb = np.zeros((Nt_max+1, 3))

    for i in range(1, Nt):
        kb[i] = 2.0 * np.cross(tang[i-1], tang[i]) / (1.0 + np.dot(tang[i-1], tang[i]))

    return kb

@njit(fastmath=True)
def computeTangents(Nt, Nt_max, ed):
    tang = np.zeros((Nt_max, 3))

    ed_norms = np.array([norm(x) for x in ed])

    for i in range(Nt):
        tang[i] = ed[i] / ed_norms[i]

    return tang

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
        K[i] = sign * 0.5 * np.dot((M[i-1] + M[i]), kb[i])

    return K



###########################################################
# PARTIAL DERIVATIVES
###########################################################

@njit(fastmath=True)
def computedKde(Nt, Nt_max, M, kb, tan, ed, inEdge, sign):
    K = computeK(Nt, Nt_max, M, kb, sign)
    Ttilda = np.zeros((Nt_max+1, 3))
    Mtilda = np.zeros((Nt_max+1, 3))
    dKde = np.zeros((Nt_max+1, 3))

    ed_norms = np.array([norm(x) for x in ed])

    for i in range(1, Nt):
        Ttilda[i] = (tan[i-1] + tan[i]) / (1.0 + np.dot(tan[i-1], tan[i]))
        Mtilda[i] = (M[i-1]   + M[i]  ) / (1.0 + np.dot(tan[i-1], tan[i]))

        if inEdge:
            dKde[i] = 1.0 / ed_norms[i]   * (-K[i] * Ttilda[i] - sign * np.cross(tan[i-1], Mtilda[i]))
        else:
            dKde[i] = 1.0 / ed_norms[i-1] * (-K[i] * Ttilda[i] + sign * np.cross(tan[i],   Mtilda[i]))

    return dKde

@njit(fastmath=True)
def computedEdm(Nt, Nt_max, Mtwist, lv, Mtwist_eq):
    dEdm = np.zeros(Nt_max+1)

    for i in range(1, Nt):
        dEdm[i] = 1.0 / lv[i] * (Mtwist[i] - Mtwist_eq[i])

    return dEdm

@njit(fastmath=True)
def computedEdK(Nt, Nt_max, M, lv, kb, Keq, sign):
    dEdK = np.zeros(Nt_max+1)
    K = computeK(Nt, Nt_max, M, kb, sign)

    for i in range(1, Nt):
        dEdK[i] = 1.0 / lv[i] * (K[i] - Keq[i])

    return dEdK

