import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from jax import jit, vmap, lax, Array
from jax.debug import print as jaxprint
from functools import partial, reduce
import jax.numpy as jnp

def aeran_model(n_i:Array, N_i:Array, lnN_i:Array, sigma_i: Array, shape: Tuple)->Array:

    Di = jnp.zeros(shape=shape)
    nip1_tot = jnp.zeros(shape=shape)
    nip1_tot.at[0].set(n_i[0])

    for i, (ni, Ni, lnNi, s_i) in enumerate(zip(n_i, N_i, lnN_i, sigma_i)):
        lnNi = jnp.log(Ni)
        # jaxprint("vals => ni, Ni, lnNi, s_i {ni}, {Ni}, {lnNi}, {s_i}",  ni = ni, Ni = Ni, lnNi = lnNi, s_i = s_i)
        delta_i = 1.25 / lnNi
        # delta_i = 1
        if i>0:
            inner_exp = sigma_i[i-1]/s_i
            # mu_i = jnp.power(inner_exp, 2)
            mu_i = inner_exp ** 2.0
            exp_ = mu_i/delta_i
            # inner = jnp.power(1.0-Di[i-1], exp_)
            inner = (1.0-Di[i-1]) ** exp_

            nip1_eff = (1.0-inner) * Ni
            nip1_tot.at[i].set(nip1_eff + ni)
            # jaxprint("niNi {Di} nip1_eff {delta} neff_i+1 {nip1}", Di=ni/Ni, delta=nip1_eff, nip1=nip1_tot[i])
            # jaxprint("Di: {Di} nip1_eff: {delta} inner: {inner}", Di=Di[i-1], delta=nip1_eff, inner=inner)

        inner_d = 1 - jnp.power((1-nip1_tot[i]/Ni), delta_i)
        Di.at[i].set(inner_d)

        # jaxprint("DAMAGE {Di} DELTA {delta} INNER _D {inner_d}", Di=Di[i], delta=delta_i, inner_d=inner_d)
        # jaxprint("niNi {Di} delta {delta} neff_i+1 {nip1}", Di=ni/Ni, delta=delta_i, nip1=nip1_tot[-1])

        # if d:=(jnp.abs(Di[i]) > 1).any():
        #     D.at[i].set(d)
        #     return jnp.array(D)
        # else:
        # D.at[i].set(jnp.abs(Di[i]))
        #
        # jaxprint("TOTAL DAMAGE => {D} // PARTIAL DAMAGE => {Di}", D=D[i], Di=Di[i])

    return jnp.abs(Di[i])
