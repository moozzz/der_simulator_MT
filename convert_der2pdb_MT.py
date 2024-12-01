#!/usr/bin/env python

import sys
import numpy as np

# import node coordinates
traj_vert = sys.argv[1]
if traj_vert == "-h":
    print('\n# Usage: ./convert_der2pdb_MT.py traj_vert.npy traj_dir.npy\n')
    sys.exit()
traj_dir = sys.argv[2]

R = np.load(traj_vert)
RD = np.load(traj_dir)

print(R.shape)
print(RD.shape)

N_frames = int(R.shape[0])
N_PF = int(R.shape[1])
Nt_array = np.zeros(N_PF, dtype=np.int8)
for p in range(N_PF):
    Nt_array[p] = int(np.count_nonzero(R[0, p, :, 0]) - 1)

print(Nt_array)

# output pdb file
fh = open('PDB_%s_%s.pdb' % (traj_vert[:-4], traj_dir[:-4]), 'w')
for i in range(N_frames):
    atom_count = 1
    fh.write('CRYST1  500.000  500.000  500.000  90.00  90.00  90.00 P 1           1\n')
    fh.write('MODEL   {:6d}\n'.format(i+1))

    for p in range(N_PF):
        for j in range(Nt_array[p]):
            if j % 2 == 0:
                fh.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format('ATOM', j+1, 'CA',
                                                                                                                                           ' ', 'MET', 'A', 1, ' ',
                                                                                                                                           (R[i, p, j, 0] + R[i, p, j+1, 0])/2.0 + 250.0,
                                                                                                                                           (R[i, p, j, 1] + R[i, p, j+1, 1])/2.0 + 250.0,
                                                                                                                                           (R[i, p, j, 2] + R[i, p, j+1, 2])/2.0 + 250.0,
                                                                                                                                           1.00, 0.00, 'C', ''))
            else:
                fh.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format('ATOM', j+1, 'CA',
                                                                                                                                           ' ', 'MET', 'B', 1, ' ',
                                                                                                                                           (R[i, p, j, 0] + R[i, p, j+1, 0])/2.0 + 250.0,
                                                                                                                                           (R[i, p, j, 1] + R[i, p, j+1, 1])/2.0 + 250.0,
                                                                                                                                           (R[i, p, j, 2] + R[i, p, j+1, 2])/2.0 + 250.0,
                                                                                                                                           1.00, 0.00, 'C', ''))

            atom_count += 1

        for j in range(Nt_array[p]):
            fh.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format('ATOM', atom_count+j, 'H ',
                                                                                                                                       ' ', 'MET', 'D', 1, ' ',
                                                                                                                                       RD[i, p, j, 0] + 250.0,
                                                                                                                                       RD[i, p, j, 1] + 250.0,
                                                                                                                                       RD[i, p, j, 2] + 250.0,
                                                                                                                                       1.00, 0.00, 'H', ''))

    fh.write('ENDMDL\n')
fh.close()

