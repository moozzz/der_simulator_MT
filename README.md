# A discrete elastic model of microtubule plus-end dynamics

### Please cite the following article when using the code:

Kalutskii, M., H. Grubmüller, V.A. Volkov and M. Igaev (2024). *Microtubule dynamics are defined by conformations and stability of clustered protofilaments*. bioRxiv [https://doi.org/10.1101/2024.11.04.621893](https://doi.org/10.1101/2024.11.04.621893)

### Content:

* `der_simulator_MT.py` -- the main script for launching a MT end simulation

* `params_bd_ff.py` -- parameters of the BD run

* `functions_pf.py` -- supplementary functions defining the PF kinematics

* `functions_forces.py` -- supplementary functions defining the forces on nodes and edge rotation angles

* `functions_bd_mt.py` -- supplementary functions defining the BD engine

* `convert_der2pdb_MT.py` -- the script for converting python `.npy` trajectories into PDB files

### Dependencies:

* `python>=3.9`

* `numpy>=1.26`

* `numba>=0.6`

* `npy-append-array>=0.9`

### Parameters of the BD run:

* `chain` -- simulation index (useful for labeling replicas when running multiple copies in parallel)

* `n_sim` -- number of restarts in a chain of simulations (useful for saving checkpoints when running
             running simulations on a cluster with a limited job time)

* `nt` -- number of steps

* `nt_skip` -- only save every `nt_skip`-th frame

* `restart_flag` -- flag for restarting a simulation from the last frame (empty or `-r`)

* `Nt_array` -- array with the number of tubulin monomers `Nt` in each PF (`Nt/2` dimers)

* `npf` -- number of PFs

* `Nt_max` -- length of the longest PF

* `Nt_frozen` -- number of frozen tubulin monomers at the minus-end

* `kbt` -- Boltzmann constant times temperature

* `nuc_state` -- nucleotide state (`gtp` or `gdp`)

* `mode_long` -- type of the longitudinal bond potential. 0.0 = harmonic (unbreakable),
                 1.0 = morse (breakable)

* `alpha_lat` -- scaling factor for the lateral interaction energy `Ulat`

* `params_diff` -- rotational and translational diffusion constants

* `params_means` -- mean edge lengths, curvatures and twists derived directly from MD

* `params_ener` -- stiffness and energy parameters optimized with Fuzzy PSO

### How to run a single simulation:

* Example: do `./der_simulator_MT.py` to run 1 simulation with index 0 of the GDP-MT end of length 12
  monomers (6 dimers) and `Ulat`*0.0 lateral interactions for 20000 steps (200 ns; each step is 10 ps)
  sampled every 2500 steps (25 ns). Adjust `params_bd_ff.py` to select a different MT geometry,
  nucleotide state and/or energy parameters.

### How to continue a single simulation:

* Example: set `restart_flag = '-r'` and `n_sim = 10` in `params_bd_run.py` to continue the above simulation
  from the last frame in a chain of 10 cycles. The trajectory of each cycle is appended to the previous one.
  Alternatively, delete the simulation directory and set `restart_flag = ''` to run a completely new
  simulation.

### How to analyze the trajectory:

* do `./convert_der2pdb_MT.py traj_vert.npy traj_dir.npy` to convert python `.npy` trajectories into
  PDB files

* use your favorite analysis tools such as `gromacs`, `MDAnalysis`, `VMD`, etc to analyze the PDB
  trajectories.

