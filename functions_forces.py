
###############################################
# IMPORT MODULES
###############################################

import numpy as np
from numpy.linalg import norm as norm
from numba import njit

from functions_pf import parallel_transport
from functions_pf import computeEdges
from functions_pf import computeTangents
from functions_pf import computeBishopFrame
from functions_pf import computeMaterialFrame
from functions_pf import computeVoronoiLen
from functions_pf import computeCurvatureBinormals
from functions_pf import computeTwist
from functions_pf import computeK1
from functions_pf import computeK2

from functions_pf import computedK1de
from functions_pf import computedK2de
from functions_pf import computedEdm
from functions_pf import computedEdK1
from functions_pf import computedEdK2



###############################################
# USER-DEFINED FORCE FUNCTIONS (PROTOFILAMENT)
###############################################

@njit(fastmath=True)
def Fstretch_diss(Nt, ed, tan, ht, Es, epsilon, a):
    Fstr = np.zeros((Nt+1, 3))
    Fs = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        if i % 2 != 0:
            # intra dimer
            Fstr[i] = Es[i]*(norm(ed[i])/ht[i] - 1.0)*tan[i]
        else:
            # inter dimer
            Fstr[i] = 2.0*epsilon*a*np.exp( -a*(norm(ed[i]) - ht[i]) ) * (1.0 - np.exp( -a*(norm(ed[i]) - ht[i]) )) * tan[i]

    for i in range(1, Nt):
        Fs[i] = Fstr[i] - Fstr[i-1]

    Fs[Nt] = -Es[Nt-1]*(norm(ed[Nt-1])/ht[Nt-1] - 1.0)*tan[Nt-1]

    return Fs

@njit(fastmath=True)
def Fstretch(Nt, ed, tan, ht, Es):
    Fstr = np.zeros((Nt+1, 3))
    Fs = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        Fstr[i] = Es[i]*(norm(ed[i])/ht[i] - 1.0)*tan[i]

    for i in range(1, Nt):
        Fs[i] = Fstr[i] - Fstr[i-1]
    
    Fs[Nt] = -Es[Nt-1]*(norm(ed[Nt-1])/ht[Nt-1] - 1.0)*tan[Nt-1]

    return Fs

@njit(fastmath=True)
def Fbend(Nt, M1, M2, kb, tan, ed, lv, K1eq, K2eq, Ek1, Ek2):
    dK1de = computedK1de(Nt, M2, kb, tan, ed, True)
    dK2de = computedK2de(Nt, M1, kb, tan, ed, True)

    dK1de_1 = computedK1de(Nt, M2, kb, tan, ed, False)
    dK2de_1 = computedK2de(Nt, M1, kb, tan, ed, False)

    K1 = computeK1(Nt, M2, kb)
    K2 = computeK2(Nt, M1, kb)

    Fb = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        dEde = Ek1[i]/lv[i] * ( (K1[i] - K1eq[i]) * dK1de[i] + (K1[i+1] - K1eq[i+1]) * dK1de_1[i+1] ) +\
               Ek2[i]/lv[i] * ( (K2[i] - K2eq[i]) * dK2de[i] + (K2[i+1] - K2eq[i+1]) * dK2de_1[i+1] )

        dEde_1 = Ek1[i]/lv[i] * ( (K1[i-1] - K1eq[i-1]) * dK1de[i-1] + (K1[i] - K1eq[i]) * dK1de_1[i] ) +\
                 Ek2[i]/lv[i] * ( (K2[i-1] - K2eq[i-1]) * dK2de[i-1] + (K2[i] - K2eq[i]) * dK2de_1[i] )

        Fb[i] = -dEde_1 +  dEde

    Fb[Nt] = -Ek1[Nt-1]/lv[Nt-1] * (K1[Nt-1] - K1eq[Nt-1]) * dK1de[Nt-1] -\
              Ek2[Nt-1]/lv[Nt-1] * (K2[Nt-1] - K2eq[Nt-1]) * dK2de[Nt-1]

    return Fb

@njit(fastmath=True)
def Ftwist(Nt, ed, Mtwist, kb, lv, Mtwist_eq, Et):
    Ft = np.zeros((Nt+1, 3))
    dEdm = computedEdm(Nt, Mtwist, lv, Mtwist_eq)

    for i in range(1, Nt):
        Ft[i] = Et[i]*dEdm[i] * ( 1.0/(2.0*norm(ed[i])) - 1.0/(2.0*norm(ed[i-1])) ) * kb[i] -\
                Et[i-1]*dEdm[i-1] * 1.0/(2.0*norm(ed[i-1])) * kb[i-1] +\
                Et[i+1]*dEdm[i+1] * 1.0/(2.0*norm(ed[i])) * kb[i+1]

    Ft[Nt] = -Et[Nt-1]*dEdm[Nt-1] * ( 1.0/(2.0*norm(ed[Nt-1])) ) * kb[Nt-1]

    return Ft

@njit(fastmath=True)
def Ftwist_theta(Nt, Mtwist, lv, Mtwist_eq, Et):
    Mt = np.zeros(Nt+1)
    dEdm = computedEdm(Nt, Mtwist, lv, Mtwist_eq)

    for i in range(1, Nt-1):
        Mt[i] = Et[i+1]*dEdm[i+1] - Et[i]*dEdm[i]

    Mt[Nt-1] = -Et[Nt-1]*dEdm[Nt-1]

    return Mt

@njit(fastmath=True)
def FcoupleM_k2(Nt, Mtwist, kb, lv, tan, ed, M1, Mtwist_eq, K2eq, Etb2):
    dEdK2 = computedEdK2(Nt, M1, lv, kb, K2eq)
    dK2de = computedK2de(Nt, M1, kb, tan, ed, True)
    dK2de_1 = computedK2de(Nt, M1, kb, tan, ed, False)
    dEdm = computedEdm(Nt, Mtwist, lv, Mtwist_eq)

    Ftb = np.zeros((Nt+1, 3))

    for i in range(1, Nt):
        dEde = Etb2[i] * dEdm[i] * dK2de[i] +\
               Etb2[i+1] * dEdm[i+1] * dK2de_1[i+1]

        dEde_1 = Etb2[i-1] * dEdm[i-1] * dK2de[i-1] +\
                 Etb2[i] * dEdm[i] * dK2de_1[i]

        Ftb[i] = Etb2[i]*dEdK2[i] * ( 1.0/(2.0*norm(ed[i])) - 1/(2*norm(ed[i-1])) ) * kb[i] -\
                 Etb2[i-1]*dEdK2[i-1] * 1.0/(2.0*norm(ed[i-1])) * kb[i-1] +\
                 Etb2[i+1]*dEdK2[i+1] * 1.0/(2.0*norm(ed[i])) * kb[i+1] -\
                 dEde_1 + dEde

    Ftb[Nt] =  -Etb2[Nt-1] * (dEdK2[Nt-1] * ( 1.0/(2.0*norm(ed[Nt-1])) ) * kb[Nt-1] + dEdm[Nt-1] * dK2de[Nt-1])

    return Ftb

@njit(fastmath=True)
def Fcouple_theta2(Nt, M1, lv, kb, K2eq, Etb2):
    dEdK2 = computedEdK2(Nt, M1, lv, kb, K2eq)
    FtbTheta = np.zeros(Nt+1)

    for i in range(1, Nt-1):
        FtbTheta[i] = Etb2[i+1]*dEdK2[i+1] - Etb2[i]*dEdK2[i]

    FtbTheta[Nt-1] = -Etb2[Nt-1]*dEdK2[Nt-1]

    return FtbTheta



###############################################
# USER-DEFINED FORCE FUNCTIONS (LATERAL)
###############################################

@njit(fastmath=True)
def get_coord(vec, m1, m2, t):
    e1 = m1*vec[0]
    e2 = m2*vec[1]
    e3 = t*vec[2]

    return e1 + e2 + e3

@njit(fastmath=True)
def rep_flat(v, v1, R0_COM_COM):
    # repulsive lateral force b/w nodes from LJ
    # eps_rep - repulsive energy per node
    # R0_COM_COM - eq distance b/w nodes

    eps_rep = 40.0 # kJ/mol
    sig_rep = 4.4578 # nm

    r = v1 - v
    r_unit = r/norm(r)

    if norm(r) < R0_COM_COM:
        rfl = 12.0 * eps_rep * ( sig_rep**12.0 / norm(r)**13.0 ) * r_unit
    else:
        rfl = np.zeros(3)

    return rfl

@njit(fastmath=True)
def attr_flat(par_epsilon_lat, R, R_unit, a_lat, R0_BS):
    # attractive lateral force from Morse
    # par_epsilon_lat - lateral energy per node
    # R0_BS - eq distance b/w binding sites

    if R >= R0_BS:
        afl =  2.0 * par_epsilon_lat * a_lat * np.exp( -a_lat*(R - R0_BS) ) * (1.0 - np.exp( -a_lat*(R - R0_BS) )) * R_unit
    else:
        afl = np.zeros(3)
    
    return afl

@njit(fastmath=True)
def Flat(Nt, M1, M2, v, tang, npf, epsilon_lat_homo, epsilon_lat_seam, a_lat_homo, a_lat_seam, alpha):
    # joint repulsive + attractive lateral force
    # alpha - scaling factor for lateral interactions
    # epsilon_lat - full lateral energy per dimer (3 nodes)
    # R0_BS_homo - eq distance b/w binding sites for homotypic contacts
    # R0_BS_seam - eq distance b/w binding sites for seam contact

    Fl = np.zeros((npf, Nt+1, 3))

    R0_COM_COM = 5.340502 # nm, eq com-com distance b/w neighbor monomers
    R0_BS_homo = 1.34050242253106 # nm, eq distance b/w homotypic binding sites
    R0_BS_seam = 1.85888211       # nm, eq distance b/w seam binding sites
    R1 =  np.array([ 0.44504187, -1.94985582, 0.0    ]) # eq coordinate of left binding site
    R2 = -np.array([ 0.44504187, -1.94985582, 0.8845 ]) # eq coordinate of shifted right binding site

    for p in range(npf):
        for i in range(1, Nt+1):
            if p == npf-1:
                # SEAM
                rr1 = get_coord(R1, M1[p, i-1], M2[p, i-1], tang[p, i-1])
                rr2 = get_coord(R2, M1[0, i+2], M2[0, i+2], tang[0, i+2])
            
                rr1_unit = rr1/norm(rr1)
                rr2_unit = rr2/norm(rr2)
            
                r1 = v[p,   i] + rr1
                r2 = v[0, i+3] + rr2

                r = r1 - r2
                r_unit = r/norm(r)
            
                fattr = attr_flat(alpha*epsilon_lat_seam/3.0, norm(r), r_unit, a_lat_seam, R0_BS_seam)
                frep =  rep_flat(v[p, i], v[0, i+3], R0_COM_COM)
            
                Fl[p, i] += -fattr - frep
                Fl[0, i] +=  fattr + frep
            else:
                # HOMOTYPIC
                rr1 = get_coord(R1, M1[p,   i-1], M2[p,   i-1], tang[p,   i-1])
                rr2 = get_coord(R2, M1[p+1, i-1], M2[p+1, i-1], tang[p+1, i-1])
                
                rr1_unit = rr1/norm(rr1)
                rr2_unit = rr2/norm(rr2)
                
                r1 = v[p,   i] + rr1
                r2 = v[p+1, i] + rr2
                
                r = r1 - r2
                r_unit = r/norm(r)
                
                fattr = attr_flat(alpha*epsilon_lat_homo/3.0, norm(r), r_unit, a_lat_homo, R0_BS_homo)
                frep =  rep_flat(v[p, i], v[p+1, i], R0_COM_COM)
                
                Fl[p,   i] += -fattr - frep
                Fl[p+1, i] +=  fattr + frep

    return Fl

