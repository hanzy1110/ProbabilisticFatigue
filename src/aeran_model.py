import numpy as np
import matplotlib.pyplot as plt
from typing import List
from jax import jit, vmap, lax, Array
from functools import partial, reduce
import jax.numpy as jnp

def aeran_model(n_i:Array, N_i:Array, lnN_i:Array, sigma_i: Array)->Array:

    Di = np.zeros_like(n_i)
    D = np.zeros_like(n_i)
    nip1_tot = np.zeros_like(n_i)
    nip1_tot[0] = n_i[0]

    for i, (ni, Ni, lnNi, s_i) in enumerate(zip(n_i, N_i, lnN_i, sigma_i)):
        delta_i = -1.25 / lnNi
        if i>0:
            mu_i = jnp.power(sigma_i[i-1]/s_i, 2)
            exp_ = mu_i/delta_i
            nip1_eff = (1-(1-jnp.power(Di[i-1], exp_))) * Ni
            nip1_tot[i] = nip1_eff + ni

        Di[i] = 1 - jnp.power((1-nip1_tot[i]/Ni), delta_i)

        if d:=jnp.abs(Di[i]) > 1:
            D[i] = d
            return jnp.array(D)
        else:
            D[i] = d


    return jnp.array(D)
