# A discrete elastic model of microtubule plus-end dynamics

Content:

- der_simulator_MT.py -- main script for launching a MT end simulation

- functions_pf.py -- supplementary functions defining PF kinematics

- functions_forces.py -- supplementary functions defining forces

- functions_bd_mt.py -- supplementary functions defining BD engine

- convert_der2pdb_MT.py -- script for converting python trajectories
                           into PDB files



How to run a single simulation:

- do './der_simulator_MT.py -h' to see the available options

- do './der_simulator_MT.py 0 gdp 1 20000 2500 12 0.0' to run
  1 simulation with index 0 of the GDP-MT end of length 12
  monomers (6 dimers) and Ulat*0.0 lateral interactions for
  20000 steps (each step is 10 ps) sampled every 2500 steps
  (25 ns)

- do './der_simulator_MT.py 0 gdp 1 20000 2500 12 0.0 -r' to
  continue the above simulation from the last frame.
  Alternatively, delete the simulation directory and do the
  command above without '-r' to run a completely new simulation.

- do './convert_der2pdb_MT.py traj_vert.npy traj_dir.npy' to
  convert python trajectories into PDB files

- the 'index' option is needed label simulations when running
  multiple copies in parallel

- the 'number of restarts' option is needed to avoid data loss
  when running simulations on clusters with limited job time

- the 'alpha' option is needed to control lateral interaction
  strength relative to its MD value

