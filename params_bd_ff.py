import sys
import numpy as np



#################################################################
# BD RUN PARAMETERS (FEEL FREE TO CHANGE!)
#################################################################

# Boltzmann constant and temperature
# assuming T ~= 300 K
kbt = 2.5 # kJ/mol

# numpy rng seed for the main time cycle
# 0 = seed assigned automatically, any integer = seed assigned manually
random_seed = 111

# simulation index
# useful when running multiple copies in parallel
chain = 0

# number of restarts in a chain of simulations
# useful for saving checkpoints when running
# simulations on a cluster with a limited job time
n_sim = 1

# number of steps in a single simulation
nt = 20000

# save trajectory every nt_skip steps
nt_skip = 2500

# restart flag (empty or '-r')
flag_restart = ''

# trajectory folder name
folder_save = 'sim_mt_%d' % chain





#################################################################
# MT PARAMETERS (FEEL FREE TO CHANGE!)
#################################################################

# array with the number of tubulin monomers Nt in each PF (Nt/2 dimers)
Nt_array = np.array([12, 10, 16, 10, 10, 10, 6, 18, 10, 8, 8, 6, 4, 10])

# number of frozen tubulin monomers at the minus-end
Nt_frozen = 1

# nucleotide state ('gtp' or 'gdp')
nuc_state = 'gdp'

# type of longitudinal bond potentials
# 0.0 = harmonic (unbreakable), 1.0 = morse (breakable)
# if harmonic, 'alpha Es' and 'alpha ht' will be used (see below)
mode_long_bond = 1.0

# scaling factor for longitudinal bond energies
# only makes sense when mode_long_bond = 1.0
# otherwise it will be ignored
alpha_long_bond = 1.0

# scaling factor for lateral bond energies
alpha_lat_bond = 1.0





#################################################################
# MT FORCE FIELD PARAMETERS (CHANGE AT YOUR OWN RISK!)
#################################################################

# rotational and translational diffusion constants
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
                              24.0*2.5,              #18 epsilon_long_bond, kJ/mol
                              3.650588936,           #19 a_long_bond, shape, 1/nm
                              mode_long_bond,        #20 0 = harmonic, 1 = morse
                              alpha_long_bond,       #21 scaling factor for longitudinal energies
                              28.729752,             #22 epsilon_lat_bond_homo, kJ/mol
                              21.021939,             #23 epsilon_lat_bond_seam, kJ/mol
                              2.061726,              #24 a_lat_bond_homo, shape, 1/nm
                              1.528234,              #25 a_lat_bond_seam, shape, 1/nm
                              alpha_lat_bond         #26 scaling factor for lateral energies
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
                              24.0*2.5,              #18 epsilon_long_bond, kJ/mol
                              4.806408919,           #19 a_long_bond, shape, 1/nm
                              mode_long_bond,        #20 0 = harmonic, 1 = morse
                              alpha_long_bond,       #21 scaling factors for longitudinal energies
                              42.853914,             #22 epsilon_lat_bond_homo, kJ/mol
                              25.857480,             #23 epsilon_lat_bond_seam, kJ/mol
                              2.164717,              #24 a_lat_bond_homo, shape, 1/nm
                              2.363691,              #25 a_lat_bond_seam, shape, 1/nm
                              alpha_lat_bond         #26 scaling factor for lateral energies
                            ])
else:
    print('\nNucleotide state can be either GTP or GDP!\n')
    sys.exit()
