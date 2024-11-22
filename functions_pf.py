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
def normalized(vec):
    return vec / norm(vec)

@njit(fastmath=True)
def parallel_transport(u, t1, t2):
    b = np.cross(t1, t2)

    if norm(b) == 0.0:
        return u

    b = normalized(b)
    n1 = np.cross(t1, b)
    n2 = np.cross(t2, b)

    return np.dot(u, t1)*t2 + np.dot(u, n1)*n2 + np.dot(u, b)*b

@njit(fastmath=True)
def computeEdges(Nt, v):
    ed = np.zeros((Nt, 3))

    for i in range(Nt):
        ed[i] = v[i+1] - v[i]

    return ed

@njit(fastmath=True)
def computeBishopFrame(Nt, t0, u0, tang):
    u = u0
    U = np.zeros((Nt, 3))
    V = np.zeros((Nt, 3))

    for i in range(Nt):
        t1 = tang[i]
        u = parallel_transport(u, t0, t1)
        u = normalized(u)
        v = np.cross(t1, u)
        U[i] = u
        V[i] = v
        t0 = t1

    return U, V

@njit(fastmath=True)
def computeMaterialFrame(Nt, U, V, theta):
    M1 = np.zeros((Nt, 3))
    M2 = np.zeros((Nt, 3))

    for i in range(Nt):
        c = np.cos(theta[i])
        s = np.sin(theta[i])
        M1[i] = c*U[i] + s*V[i]
        M2[i] = -s*U[i] + c*V[i]

    return M1, M2

@njit(fastmath=True)
def computeVoronoiLen(Nt, ed):
    lv = np.zeros(Nt+1)
    lv[0]  = 0.5 * norm(ed[0])
    lv[-1] = 0.5 * norm(ed[-1])

    for i in range(1, Nt):
        lv[i] = 0.5 * (norm(ed[i-1]) + norm(ed[i]))

    return lv

@njit(fastmath=True)
def computeCurvatureBinormals(Nt, tang):
    kb = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        t1 = tang[i-1]
        t2 = tang[i]
        kb[i] = 2.0 * np.cross(t1, t2) / (1.0 + np.dot(t1, t2))

    return kb

@njit(fastmath=True)
def computeTangents(Nt, ed):
    tang = np.zeros((Nt, 3))

    for i in range(Nt):
        tang[i] = normalized(ed[i])

    return tang

@njit(fastmath=True)
def computeTwist(Nt, theta, mref):
    Mtwist = np.zeros(Nt+1)

    for i in range(1, Nt):
        Mtwist[i] = theta[i] - theta[i-1] + mref[i]

    return Mtwist

@njit(fastmath=True)
def computeK1(Nt, M2, kb):
    K1 = np.zeros(Nt+1)

    for i in range(1, Nt):
        K1[i] = 0.5 * np.dot((M2[i-1] + M2[i]), kb[i])

    return K1

@njit(fastmath=True)
def computeK2(Nt, M1, kb):
    K2 = np.zeros(Nt+1)

    for i in range(1, Nt):
        K2[i] = -0.5 * np.dot((M1[i-1] + M1[i]), kb[i])

    return K2



###########################################################
# PARTIAL DERIVATIVES
###########################################################

@njit(fastmath=True)
def computedK1de(Nt, M2, kb, tan, ed, inEdge):
    K1 = computeK1(Nt, M2, kb)
    Ttilda = np.zeros((Nt+1, 3))
    Mtilda = np.zeros((Nt+1, 3))
    dK1de = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        Ttilda[i] = (tan[i-1] + tan[i]) / (1.0 + np.dot(tan[i-1], tan[i]))
        Mtilda[i] = (M2[i-1] + M2[i]) / (1.0 + np.dot(tan[i-1], tan[i]))

        if inEdge:
            dK1de[i] = 1.0/norm(ed[i]) * ( -K1[i]*Ttilda[i] - np.cross(tan[i-1], Mtilda[i]) )
        else:
            dK1de[i] = 1.0/norm(ed[i-1]) * ( -K1[i]*Ttilda[i] + np.cross(tan[i], Mtilda[i]) )

    return dK1de

@njit(fastmath=True)
def computedK2de(Nt, M1, kb, tan, ed, inEdge):
    K2 = computeK2(Nt, M1, kb)
    Ttilda = np.zeros((Nt+1, 3))
    Mtilda = np.zeros((Nt+1, 3))
    dK2de = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        Ttilda[i] = (tan[i-1] + tan[i]) / (1.0 + np.dot(tan[i-1], tan[i]))
        Mtilda[i] = (M1[i-1] + M1[i]) / (1.0 + np.dot(tan[i-1], tan[i]))

        if inEdge:
            dK2de[i] = 1.0/norm(ed[i]) * ( -K2[i]*Ttilda[i] + np.cross(tan[i-1], Mtilda[i]) )
        else:
            dK2de[i] = 1.0/norm(ed[i-1]) * ( -K2[i]*Ttilda[i] - np.cross(tan[i], Mtilda[i]) )

    return dK2de

@njit(fastmath=True)
def computedEdm(Nt, Mtwist, lv, Mtwist_eq):
    dEdm = np.zeros(Nt+1)

    for i in range(1, Nt):
        dEdm[i] = 1.0/lv[i]*(Mtwist[i] - Mtwist_eq[i])

    return dEdm

@njit(fastmath=True)
def computedEdK1(Nt, M2, lv, kb, K1eq):
    dEdK1 = np.zeros(Nt+1)
    K1 = computeK1(Nt, M2, kb)

    for i in range(1, Nt):
        dEdK1[i] = 1.0/lv[i]*(K1[i] - K1eq[i])

    return dEdK1

@njit(fastmath=True)
def computedEdK2(Nt, M1, lv, kb, K2eq):
    dEdK2 = np.zeros(Nt+1)
    K2 = computeK2(Nt, M1, kb)

    for i in range(1, Nt):
        dEdK2[i] = 1.0/lv[i]*(K2[i] - K2eq[i])

    return dEdK2

