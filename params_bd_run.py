# simulation index
chain = 0

# nucleotide state ('gtp' or 'gdp')
nuc_state = 'gdp'

# number of restarts in a chain of simulations
n_sim = 1

# number of steps in a single simulation
nt = 20000

# save trajectory every nt_skip steps
nt_skip = 2500

# number of monomers in each PF
Nt_array = [12, 10, 16, 10, 10, 10, 6, 18, 10, 8, 8, 6, 4, 10]

# scaling factor for lateral energies
alpha = 1.0

# restart flag (empty or '-r')
restart_flag = ''
