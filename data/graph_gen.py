#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as md
from p2g import pos2graph

r_c_arr = np.array([2.5])
f_c_arr = np.array([0.55])

gap = 100
T_min = 100
T_max = 900+gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype = "float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
t_arr = T

p_arr = np.array([0,1,2,3,4,5,6,7,8,9])

set_arr = np.array([1,2])

for r_c in r_c_arr:
    for f_c in f_c_arr:
        for t in t_arr:
            for p in p_arr:
                for set in set_arr:
                    source = "Traj/Set_" + str(int(set)) + "/T_" + str(int(p)) + "/"
                    traj_file = source + "traj.dcd"
                    u = md.lib.formats.libdcd.DCDFile(traj_file)
                    N_frame = u.n_frames
                    n_atom = u.header['natoms']
                    traj = u.readframes()[0]
                    pos = traj[int(t)]
                    A = pos2graph(pos,r_c,f_c)
                    data_save = "./graph/set_" + str(set) + "/D_" + str(int(p)) + "/T_" + str(int(t)) + "_r_" + str(r_c) + "_f_" + str(f_c) + ".npy"
                    np.save(data_save, A, allow_pickle=True)
                    print(p, "_", f_c, "_", t)



