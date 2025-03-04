#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def pos2graph(pos, r_c, f_c):
    N = len(pos)
    A = np.zeros((N,N), dtype = "int")
    for i in range(N):
        S = dict([])
        x_i = pos[i][0]
        y_i = pos[i][1]
        z_i = pos[i][2]
        for p in range(N):
            x_p = pos[p][0]
            y_p = pos[p][1]
            z_p = pos[p][2]
            r_ip = np.sqrt(np.power(x_i-x_p,2) + np.power(y_i-y_p,2) + np.power(z_i-z_p,2))
            if(r_ip < r_c):
                pr = {p : r_ip}
                S.update(pr)
        S.pop(i)
        Sl = sorted(S.items(), key = lambda kv:(kv[1], kv[0]))
        S = dict(Sl)
        for j in range(len(Sl)):
            j_idx = Sl[j][0]
            x_j = pos[j_idx][0]
            y_j = pos[j_idx][1]
            z_j = pos[j_idx][2]
            r_ij = np.sqrt(np.power(x_i-x_j,2) + np.power(y_i-y_j,2) + np.power(z_i-z_j,2))
            for k in range(j+1, len(Sl)):
                k_idx = Sl[k][0]
                x_k = pos[k_idx][0]
                y_k = pos[k_idx][1]
                z_k = pos[k_idx][2]
                r_ik = np.sqrt(np.power(x_i-x_k,2) + np.power(y_i-y_k,2) + np.power(z_i-z_k,2))
                r_jk = np.sqrt(np.power(x_j-x_k,2) + np.power(y_j-y_k,2) + np.power(z_j-z_k,2))
                ruler = np.power(r_ik,2) / (np.power(r_ij,2) + np.power(r_jk,2))
                if f_c <= ruler:
                    S.pop(k_idx,0)
        for idx, val in S.items():
            A[i,idx] = 1
            A[idx,i] = 1 # The generated A must be a symmetry matrix.
    return A

