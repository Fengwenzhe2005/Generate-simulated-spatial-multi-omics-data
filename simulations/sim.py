#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from pandas import get_dummies
from anndata import AnnData
from scanpy import pp
from utils import misc, preprocess

dtp = "float32"

random.seed(101)

def squares():
    A = np.zeros([12, 12])
    A[1:5, 1:5] = 1; A[7:11, 1:5] = 1; A[1:5, 7:11] = 1; A[7:11, 7:11] = 1
    return A

def corners():
    B = np.zeros([6, 6])
    for i in range(6): B[i, i:] = 1
    A = np.flip(B, axis=1)
    AB = np.hstack((A, B)); CD = np.flip(AB, axis=0)
    return np.vstack((AB, CD))

def scotland():
    A = np.eye(12)
    for i in range(12): A[-i-1, i] = 1
    return A

def checkers():
    A = np.zeros([4, 4]); B = np.ones([4, 4])
    AB = np.hstack((A, B, A)); BA = np.hstack((B, A, B))
    return np.vstack((AB, BA, AB))

def quilt():
    A = np.zeros([4, 144])
    A[0, :] = squares().flatten()
    A[1, :] = corners().flatten()
    A[2, :] = scotland().flatten()
    A[3, :] = checkers().flatten()
    return A

def ggblocks():
    A = np.zeros([4, 36])
    A[0, [1, 6, 7, 8, 13]] = 1
    A[1, [3, 4, 5, 9, 11, 15, 16, 17]] = 1
    A[2, [18, 24, 25, 30, 31, 32]] = 1
    A[3, [21, 22, 23, 28, 34]] = 1
    return A

def spamv():
    A = np.zeros((9, 36, 36))
    
    A[0, 0:12, 4:8] = 1
    A[0, 0:4, 0:12] = 1
    
    A[1, 0:12, 12:24] = 1
    A[1, 4:8, 16:20] = 0
    
    A[2, 0:12, 24:28] = 1
    A[2, 8:12, 24:36] = 1
    
    A[3, 12:24, 4:8] = 1
    A[3, 16:20, 0:12] = 1
    
    A[4, 12:16, 12:24] = 1
    A[4, 12:24, 12:16] = 1
    
    A[5, 12:16, 24:36] = 1
    A[5, 12:24, 28:32] = 1
    
    A[6, 24:36, 8:12] = 1
    A[6, 32:36, 0:12] = 1
    
    A[7, 32:36, 12:24] = 1
    A[7, 28:32, 16:24] = 1
    A[7, 24:28, 20:24] = 1
    
    A[8, 28:32, 24:36] = 1

    return A.reshape(9, 36*36)

def sqrt_int(x):
    z = int(round(x**.5))
    if x == z**2: return z
    else: raise ValueError("x must be a square integer")

def gen_spatial_factors(scenario="quilt", nside=36):
    if scenario == "quilt": A = quilt()
    elif scenario == "ggblocks": A = ggblocks()
    elif scenario == "spamv": A = spamv()
    else: raise ValueError("scenario must be 'quilt', 'ggblocks', or 'spamv'")
    
    unit = sqrt_int(A.shape[1])
    assert nside % unit == 0
    ncopy = nside // unit
    N = nside**2
    L = A.shape[0]
    A = A.reshape((L, unit, unit))
    A = np.kron(A, np.ones((1, ncopy, ncopy)))
    F = A.reshape((L, N)).T
    return F

def gen_spatial_coords(N):
    X = misc.make_grid(N)
    X[:, 1] = -X[:, 1]
    return preprocess.rescale_spatial_coords(X)

def gen_nonspatial_factors(N, L=3, nzprob=0.2, seed=101):
    rng = np.random.default_rng(seed)
    return rng.binomial(1, nzprob, size=(N, L))

def gen_loadings(Lsp, Lns=3, Jsp=0, Jmix=500, Jns=0, expr_mean=20.0,
                 mix_frac_spat=0.55, seed=101, **kwargs):
    rng = np.random.default_rng(seed)
    J = Jsp + Jmix + Jns
    if Lsp > 0:
        w = rng.choice(Lsp, J, replace=True)
        W = get_dummies(w).to_numpy(dtype=dtp)
    else:
        W = np.zeros((J, 0))
    if Lns > 0:
        v = rng.choice(Lns, J, replace=True)
        V = get_dummies(v).to_numpy(dtype=dtp)
    else:
        V = np.zeros((J, 0))
    
    W[:Jsp, :] *= expr_mean
    V[:Jsp, :] = 0
    W[Jsp:(Jsp+Jmix), :] *= (mix_frac_spat * expr_mean)
    V[Jsp:(Jsp+Jmix), :] *= ((1 - mix_frac_spat) * expr_mean)
    W[(Jsp+Jmix):, :] = 0
    V[(Jsp+Jmix):, :] *= expr_mean
    return W, V

def sim2anndata(locs, outcome, spfac, spload, nsfac=None, nsload=None):
    obsm = {"spatial": locs, "spfac": spfac, "nsfac": nsfac}
    varm = {"spload": spload, "nsload": nsload}
    ad = AnnData(outcome, obsm=obsm, varm=varm)
    ad.layers = {"counts": ad.X.copy()}
    pp.log1p(ad)
    idx = list(range(ad.shape[0]))
    random.shuffle(idx)
    ad = ad[idx, :]
    return ad

class SpatialMultiOmics:
    def __init__(self, scenario="quilt", nside=36, seed=101):
        self.nside = nside
        self.N = nside ** 2
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.coords = gen_spatial_coords(self.N)
        
        if scenario == "both":
            F1 = gen_spatial_factors(nside=nside, scenario="ggblocks")
            F2 = gen_spatial_factors(nside=nside, scenario="quilt")
            self.F = np.hstack((F1, F2))
        else:
            self.F = gen_spatial_factors(scenario=scenario, nside=nside)
        
        self.n_spatial_factors = self.F.shape[1]
        
        self.Lns = 3
        self.U = gen_nonspatial_factors(self.N, L=self.Lns, nzprob=0.2, seed=seed)

    def generate_data(self, modality_config):
        results = {}
        
        for mod in modality_config:
            name = mod["name"]
            params = mod.get("params", {})
            active_factors = mod.get("active_factors", None) 
            
            mod_seed = self.seed + sum(ord(c) for c in name)
            mod_rng = np.random.default_rng(mod_seed)
        
            F_mod = self.F.copy()
            if active_factors is not None:
                mask = np.zeros(self.n_spatial_factors, dtype=bool)
                for idx in active_factors:
                    if 0 <= idx < self.n_spatial_factors:
                        mask[idx] = True
                F_mod[:, ~mask] = 0
            
            W, V = gen_loadings(
                Lsp=self.n_spatial_factors,
                Lns=self.Lns,
                Jsp=params.get("Jsp", 200),
                Jmix=params.get("Jmix", 200),
                Jns=params.get("Jns", 200),
                expr_mean=params.get("expr_mean", 20.0),
                mix_frac_spat=params.get("mix_frac", 0.55),
                seed=mod_seed
            )
            
            bkg_mean = 0.2
            Lambda = bkg_mean + F_mod @ W.T + self.U @ V.T
            
            r = params.get("nb_shape", 10.0)
            p_probs = r / (Lambda + r + 1e-10)
            Y = mod_rng.negative_binomial(r, p_probs)
            

            dropout_rate = params.get("dropout_rate", 0.0)
            if dropout_rate > 0.0:
                keep_mask = mod_rng.binomial(n=1, p=(1 - dropout_rate), size=Y.shape)
                Y = Y * keep_mask
            
            ad = sim2anndata(self.coords, Y, self.F, W, nsfac=self.U, nsload=V)
            
            ad.var_names = [f"{name}_feat_{i}" for i in range(ad.shape[1])]
            ad.obs_names = [f"spot_{i}" for i in range(ad.shape[0])]
            ad.uns['modality'] = name
            ad.uns['active_factors'] = active_factors if active_factors else "all"
            
            results[name] = ad
            print(f"Generated [{name}]: shape={ad.shape}, dropout={dropout_rate}, active_factors={ad.uns['active_factors']}")
            
        return results

def sim(scenario, nside=36, nzprob_nsp=0.2, bkg_mean=0.2, nb_shape=10.0,
        seed=101, dropout_rate=0.0, **kwargs):

    simulator = SpatialMultiOmics(scenario=scenario, nside=nside, seed=seed)
    
    config = [{
        "name": "SingleModality",
        "params": {
            "nb_shape": nb_shape,
            "dropout_rate": dropout_rate,
            **kwargs
        },
        "active_factors": None
    }]
    
    results = simulator.generate_data(config)
    return results["SingleModality"]