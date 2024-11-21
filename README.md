# A discrete elastic model of microtubule plus-end dynamics

### Content:

* `der_simulator_MT.py` -- the main script for launching a MT end simulation

* `functions_pf.py` -- supplementary functions defining the PF kinematics

* `functions_forces.py` -- supplementary functions defining the forces on nodes and edge rotation angles

* `functions_bd_mt.py` -- supplementary functions defining the BD engine

* `convert_der2pdb_MT.py` -- the script for converting python `.npy` trajectories into PDB files

### Dependencies:

* `python 3.9` or newer

* `numpy 1.26` or newer

* `numba 0.6` or newer

* `npy-append-array 0.9` or newer

### Parameters:

* Usage: `./der_simulator_MT.py chain nuc_state n_sim nt nt_skip Nt alpha restart_flag`

* `chain` -- simulation index (useful for labeling replicas when running multiple copies in parallel)

* `nuc_state` -- nucleotide state (`gtp` or `gdp`)

* `n_sim` -- number of restarts in a chain of simulations (useful for saving checkpoints when running
             running simulations on a cluster with a limited job time)

* `nt` -- number of steps

* `nt_skip` -- only save every `nt_skip`-th frame

* `Nt` -- number of tubulin monomers in a PF (`Nt/2` dimers)

* `alpha` -- scaling factor for the lateral interaction energy `Ulat`

* `restart_flag` -- flag for restarting a simulation from the last frame (empty or `-r`)

### How to run and/or continue a single simulation:

* do `./der_simulator_MT.py 0 gdp 1 20000 2500 12 0.0` to run 1 simulation with index 0 of the GDP-MT
  end of length 12 monomers (6 dimers) and `Ulat`*0.0 lateral interactions for 20000 steps (each step
  is 10 ps) sampled every 2500 steps (25 ns)

* do `./der_simulator_MT.py 0 gdp 1 20000 2500 12 0.0 -r` to continue the above simulation from the
  last frame. Alternatively, delete the simulation directory and do the command above without `-r` to
  run a completely new simulation.

* do `./convert_der2pdb_MT.py traj_vert.npy traj_dir.npy` to convert python `.npy` trajectories into
  PDB files
