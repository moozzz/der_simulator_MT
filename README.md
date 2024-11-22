# A discrete elastic model of microtubule plus-end dynamics

### Content:

* `der_simulator_MT.py` -- the main script for launching a MT end simulation

* `params_bd_run.py` -- parameters for the BD run

* `functions_pf.py` -- supplementary functions defining the PF kinematics

* `functions_forces.py` -- supplementary functions defining the forces on nodes and edge rotation angles

* `functions_bd_mt.py` -- supplementary functions defining the BD engine

* `convert_der2pdb_MT.py` -- the script for converting python `.npy` trajectories into PDB files

### Dependencies:

* `python>=3.9`

* `numpy>=1.26`

* `numba>=0.6`

* `npy-append-array>=0.9`

### Parameters:

* `chain` -- simulation index (useful for labeling replicas when running multiple copies in parallel)

* `nuc_state` -- nucleotide state (`gtp` or `gdp`)

* `n_sim` -- number of restarts in a chain of simulations (useful for saving checkpoints when running
             running simulations on a cluster with a limited job time)

* `nt` -- number of steps

* `nt_skip` -- only save every `nt_skip`-th frame

* `Nt` -- number of tubulin monomers in a PF (`Nt/2` dimers)

* `alpha` -- scaling factor for the lateral interaction energy `Ulat`

* `restart_flag` -- flag for restarting a simulation from the last frame (empty or `-r`)

### How to run a single simulation:

* do `./der_simulator_MT.py` to run 1 simulation with index 0 of the GDP-MT end of length 12 monomers
  (6 dimers) and `Ulat`*0.0 lateral interactions for 20000 steps (200 ns; each step is 10 ps) sampled
  every 2500 steps (25 ns)

### How to continue a single simulation:

* set `restart_flag = '-r'` and `n_sim = 10` in `params_bd_run.py` to continue the above simulation from
  the last frame in a chain of 10 cycles. The trajectory of each cycle is appended to the previous one.
  Alternatively, delete the simulation directory and set `restart_flag = ''` to run a completely new
  simulation.

### How to analyze the trajectory:

* do `./convert_der2pdb_MT.py traj_vert.npy traj_dir.npy` to convert python `.npy` trajectories into
  PDB files

* use your favorite analysis tools such as `gromacs`, `MDAnalysis`, etc to analyze the PDB trajectories.

