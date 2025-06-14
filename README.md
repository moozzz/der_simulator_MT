# A discrete elastic model of microtubule plus-end dynamics

### Please cite the following article when using the code:

Kalutskii, M., H. Grubmüller, V.A. Volkov and M. Igaev (2025). *Microtubule dynamics are defined by conformations and stability of clustered protofilaments*. [https://doi.org/10.1073/pnas.2424263122](PNAS 122(22): e2424263122)

<div align="center">
  <img src="mt_movie.gif">
</div>

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

* `npy-append-array>=0.9` ([https://github.com/xor2k/npy-append-array](https://github.com/xor2k/npy-append-array))

### Parameters of the BD run:

See the comments in `params_bd_ff.py` for further information.

### How to run a single simulation:

* Example: do `./der_simulator_MT.py` to run 1 simulation with index 0 of the GDP-MT end for 20000 steps
  (200 ns; each step is 10 ps) sampled every 2500 steps (25 ns). Adjust `params_bd_ff.py` to select a
  different MT geometry, run times, nucleotide state and/or energy parameters.

### How to continue a single simulation:

* Example: set `flag_restart = '-r'` and `n_sim = 10` in `params_bd_run.py` to continue the above simulation
  from the last frame in a chain of 10 cycles. The trajectory of each cycle is appended to the previous one.
  Alternatively, delete the simulation folder and set `flag_restart = ''` to run a completely new
  simulation.

### How to analyze the trajectory:

* do `./convert_der2pdb_MT.py traj_vert.npy traj_dir.npy` to convert python `.npy` trajectories into
  PDB files.

* use your favorite analysis tools such as `gromacs`, `MDAnalysis`, `VMD`, etc to analyze the PDB
  trajectories.

