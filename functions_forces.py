###############################################
# IMPORT MODULES
###############################################

import numpy as np
from numpy.linalg import norm as norm
from numba import njit

from functions_pf import computeK
from functions_pf import computedMde
from functions_pf import computedUdM
from functions_pf import computedKde
from functions_pf import computedUdK



###############################################
# USER-DEFINED FORCE FUNCTIONS (PROTOFILAMENT)
###############################################

@njit(fastmath=True)
def Fstretch(Nt, Nt_max, ed, tan, ht, Es, epsilon_long_bond, a_long_bond, mode_long_bond, alpha_long_bond):
    Fstr = np.zeros((Nt_max+1, 3))
    Fs = np.zeros((Nt_max+1, 3))

    ed_norms = np.array([norm(x) for x in ed])

    for i in range(1, Nt):
        Fstr[i] = Es[i] * (ed_norms[i] / ht[i] - 1.0) * tan[i]

        if mode_long_bond == 1.0 and i % 2 == 0:
            # only for inter-dimer interfaces
            Fstr[i] = 2.0 * alpha_long_bond * epsilon_long_bond * a_long_bond * np.exp( -a_long_bond * (ed_norms[i] - ht[i]) ) * ( 1.0 - np.exp( -a_long_bond * (ed_norms[i] - ht[i]) ) ) * tan[i]

    for i in range(1, Nt):
        Fs[i] = Fstr[i] - Fstr[i-1]

    Fs[Nt] = -Es[Nt-1] * (ed_norms[Nt-1] / ht[Nt-1] - 1.0) * tan[Nt-1]

    return Fs

@njit(fastmath=True)
def Fbend(Nt, Nt_max, M1, M2, kb, tan, ed, lv, K1eq, K2eq, Ek1, Ek2):
    dK1de = computedKde(Nt, Nt_max, M2, kb, tan, ed, True, 1.0)
    dK2de = computedKde(Nt, Nt_max, M1, kb, tan, ed, True, -1.0)

    dK1de_1 = computedKde(Nt, Nt_max, M2, kb, tan, ed, False, 1.0)
    dK2de_1 = computedKde(Nt, Nt_max, M1, kb, tan, ed, False, -1.0)

    K1 = computeK(Nt, Nt_max, M2, kb, 1.0)
    K2 = computeK(Nt, Nt_max, M1, kb, -1.0)

    Fb = np.zeros((Nt_max+1, 3))

    for i in range(1, Nt):
        dEde = Ek1[i] / lv[i] * ( (K1[i] - K1eq[i]) * dK1de[i] + (K1[i+1] - K1eq[i+1]) * dK1de_1[i+1] ) +\
               Ek2[i] / lv[i] * ( (K2[i] - K2eq[i]) * dK2de[i] + (K2[i+1] - K2eq[i+1]) * dK2de_1[i+1] )

        dEde_1 = Ek1[i] / lv[i] * ( (K1[i-1] - K1eq[i-1]) * dK1de[i-1] + (K1[i] - K1eq[i]) * dK1de_1[i] ) +\
                 Ek2[i] / lv[i] * ( (K2[i-1] - K2eq[i-1]) * dK2de[i-1] + (K2[i] - K2eq[i]) * dK2de_1[i] )

        Fb[i] = -dEde_1 +  dEde

    Fb[Nt] = -Ek1[Nt-1] / lv[Nt-1] * (K1[Nt-1] - K1eq[Nt-1]) * dK1de[Nt-1] -\
              Ek2[Nt-1] / lv[Nt-1] * (K2[Nt-1] - K2eq[Nt-1]) * dK2de[Nt-1]

    return Fb

@njit(fastmath=True)
def Ftwist(Nt, Nt_max, ed, Mtwist, kb, lv, Mtwist_eq, Et):
    Ft = np.zeros((Nt_max+1, 3))
    dUdM = computedUdM(Nt, Nt_max, Mtwist, lv, Mtwist_eq)

    ed_norms = np.array([norm(x) for x in ed])

    for i in range(1, Nt):
        Ft[i] = Et[i]   * dUdM[i]   * ( 1.0 / (2.0 * ed_norms[i]) - 1.0 / (2.0 * ed_norms[i-1]) ) * kb[i] -\
                Et[i-1] * dUdM[i-1] *   1.0 / (2.0 * ed_norms[i-1]) * kb[i-1] +\
                Et[i+1] * dUdM[i+1] *   1.0 / (2.0 * ed_norms[i]  ) * kb[i+1]

    Ft[Nt] = -Et[Nt-1] * dUdM[Nt-1] * ( 1.0 / (2.0 * ed_norms[Nt-1]) ) * kb[Nt-1]

    return Ft

@njit(fastmath=True)
def Ftwist_theta(Nt, Nt_max, Mtwist, lv, Mtwist_eq, Et):
    Mt = np.zeros(Nt_max+1)
    dUdM = computedUdM(Nt, Nt_max, Mtwist, lv, Mtwist_eq)

    for i in range(1, Nt-1):
        Mt[i] = Et[i+1] * dUdM[i+1] - Et[i] * dUdM[i]

    Mt[Nt-1] = -Et[Nt-1] * dUdM[Nt-1]

    return Mt

@njit(fastmath=True)
def FcoupleM_k2(Nt, Nt_max, Mtwist, kb, lv, tan, ed, M1, Mtwist_eq, K2eq, Etb2):
    dUdK2 = computedUdK(Nt, Nt_max, M1, lv, kb, K2eq, -1.0)
    dK2de = computedKde(Nt, Nt_max, M1, kb, tan, ed, True, -1.0)
    dK2de_1 = computedKde(Nt, Nt_max, M1, kb, tan, ed, False, -1.0)
    dUdM = computedUdM(Nt, Nt_max, Mtwist, lv, Mtwist_eq)

    Ftb = np.zeros((Nt_max+1, 3))

    ed_norms = np.array([norm(x) for x in ed])

    for i in range(1, Nt):
        dEde = Etb2[i] * dUdM[i] * dK2de[i] +\
               Etb2[i+1] * dUdM[i+1] * dK2de_1[i+1]

        dEde_1 = Etb2[i-1] * dUdM[i-1] * dK2de[i-1] +\
                 Etb2[i] * dUdM[i] * dK2de_1[i]

        Ftb[i] = Etb2[i]   * dUdK2[i]   * ( 1.0 / (2.0 * ed_norms[i]  ) - 1.0/(2.0 * ed_norms[i-1]) ) * kb[i] -\
                 Etb2[i-1] * dUdK2[i-1] *   1.0 / (2.0 * ed_norms[i-1]) * kb[i-1] +\
                 Etb2[i+1] * dUdK2[i+1] *   1.0 / (2.0 * ed_norms[i]  ) * kb[i+1] -\
                 dEde_1 + dEde

    Ftb[Nt] =  -Etb2[Nt-1] * (dUdK2[Nt-1] * ( 1.0 / (2.0 * ed_norms[Nt-1]) ) * kb[Nt-1] + dUdM[Nt-1] * dK2de[Nt-1])

    return Ftb

@njit(fastmath=True)
def Fcouple_theta2(Nt, Nt_max, M1, lv, kb, K2eq, Etb2):
    dUdK2 = computedUdK(Nt, Nt_max, M1, lv, kb, K2eq, -1.0)
    FtbTheta = np.zeros(Nt_max+1)

    for i in range(1, Nt-1):
        FtbTheta[i] = Etb2[i+1] * dUdK2[i+1] - Etb2[i] * dUdK2[i]

    FtbTheta[Nt-1] = -Etb2[Nt-1] * dUdK2[Nt-1]

    return FtbTheta



###############################################
# USER-DEFINED FORCE FUNCTIONS (LATERAL)
###############################################

@njit(fastmath=True)
def get_coord(vec, m1, m2, t):
    e1 = m1 * vec[0]
    e2 = m2 * vec[1]
    e3 = t  * vec[2]

    return e1 + e2 + e3

@njit(fastmath=True)
def rep_flat(v, v1, R0_COM_COM):
    # repulsive lateral force b/w nodes from LJ
    # epsilon_rep - repulsive energy per node
    # R0_COM_COM - eq distance b/w nodes

    epsilon_rep = 40.0 # kJ/mol
    sig_rep = 4.4578 # nm

    r = v1 - v
    r_unit = r / norm(r)

    if norm(r) < R0_COM_COM:
        rfl = 12.0 * epsilon_rep * ( sig_rep**12.0 / norm(r)**13.0 ) * r_unit
    else:
        rfl = np.zeros(3)

    return rfl

@njit(fastmath=True)
def attr_flat(par_epsilon_lat_bond, R, R_unit, a_lat_bond, R0_BS):
    # attractive lateral force from Morse
    # par_epsilon_lat_bond - lateral energy per node
    # R0_BS - eq distance b/w binding sites

    if R >= R0_BS:
        afl =  2.0 * par_epsilon_lat_bond * a_lat_bond * np.exp( -a_lat_bond * (R - R0_BS) ) * (1.0 - np.exp( -a_lat_bond * (R - R0_BS) )) * R_unit
    else:
        afl = np.zeros(3)
    
    return afl

@njit(fastmath=True)
def Flat(Nt_array, M1, M2, v, tang, epsilon_lat_bond_homo, epsilon_lat_bond_seam, a_lat_bond_homo, a_lat_bond_seam, alpha_lat_bond):
    # joint repulsive + attractive lateral force
    # alpha_lat_bond - scaling factor for lateral interactions
    # epsilon_lat_bond - full lateral energy per dimer (3 nodes)
    # R0_BS_homo - eq distance b/w binding sites for homotypic contacts
    # R0_BS_seam - eq distance b/w binding sites for seam contact

    npf = len(Nt_array)
    Nt_max = int(np.max(Nt_array))

    Fl = np.zeros((npf, Nt_max+1, 3))

    R0_COM_COM = 5.340502 # nm, eq com-com distance b/w neighbor monomers
    R0_BS_homo = 1.34050242253106 # nm, eq distance b/w homotypic binding sites
    R0_BS_seam = 1.85888211       # nm, eq distance b/w seam binding sites
    R1 =  np.array([ 0.44504187, -1.94985582, 0.0     ]) # eq coordinate of left binding site
    R2 = -np.array([ 0.44504187, -1.94985582, -0.8845 ]) # eq coordinate of shifted right binding site

    for p in range(npf):
        seam = (p == npf-1)
        R0_BS = R0_BS_seam if seam else R0_BS_homo
        epsilon_lat_bond = epsilon_lat_bond_seam if seam else epsilon_lat_bond_homo
        a_lat_bond = a_lat_bond_seam if seam else a_lat_bond_homo

        for i in range(1, Nt_array[p]+1):
            rr1 = get_coord(R1, M1[p,           i-1], M2[p,           i-1], tang[p,           i-1])
            rr2 = get_coord(R2, M1[(p+1) % npf, i-1], M2[(p+1) % npf, i-1], tang[(p+1) % npf, i-1])

            r1 = v[p,           i] + rr1
            r2 = v[(p+1) % npf, i] + rr2

            r = r1 - r2
            r_unit = r / norm(r)

            fattr = attr_flat(alpha_lat_bond * epsilon_lat_bond / 3.0, norm(r), r_unit, a_lat_bond, R0_BS)
            frep =  rep_flat(v[p, i], v[(p+1) % npf, i], R0_COM_COM)

            Fl[p,           i] += -fattr - frep
            Fl[(p+1) % npf, i] +=  fattr + frep

    return Fl

