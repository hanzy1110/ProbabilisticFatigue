import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from jax import jit, vmap, lax, Array
from jax.debug import print as jaxprint, breakpoint
from functools import partial, reduce
import jax.numpy as jnp

from jax import config

config.update("jax_debug_nans", True)
config.update("jax_debug_infs", True)


def linear_damage(l, Di, DiPrev, ni, Ni):
    # jaxprint("LINEAR MODEL")

    # jaxprint("lin_dmg --> {lin_dmg}", lin_dmg=ni / Ni)
    DiPrev = Di
    Di = ni / Ni
    return jnp.array([Di, DiPrev])
    # return {"Di":Di, "DiPrev":DiPrev}


def aeran_damage(l, i, Di, DiPrev, ni, Ni, mu_i, delta_i):
    # jaxprint("AERAN MODEL lin_dmg --> {lin_dmg}", lin_dmg=ni / Ni)

    nip1_tot = ni
    if i > 0:
        exp_ = mu_i / delta_i
        inner = jnp.power(jnp.abs(1.0 - DiPrev), exp_)
        nip1_eff = (1.0 - inner) * Ni
        nip1_tot = nip1_eff + ni

    inner_d = 1 - jnp.power(jnp.abs(1 - nip1_tot / Ni), delta_i)
    DiPrev = Di
    Di = inner_d
    return jnp.array([Di, DiPrev])
    # return {"Di":Di, "DiPrev":DiPrev}


# @jit
def aeran_model(n_i: Array, N_i: Array, sigma_i: Array) -> Array:
    Di = 0.0
    DiPrev = 0.0
    # nip1_tot = n_i[0]

    for i, (ni, Ni, s_i) in enumerate(zip(n_i, N_i, sigma_i)):
        lnNi = jnp.log(Ni)
        delta_i = 1.25 / lnNi
        mu_i = (sigma_i[i - 1] / s_i) ** 2.0

        lin_dmg = ni / Ni

        # dmgs = jnp.piecewise(lin_dmg, [lin_dmg>=1, lin_dmg<1],
        #                            [lambda x: linear_damage(x, Di, DiPrev, ni, Ni),
        #                             lambda x: aeran_damage(x, i, Di, DiPrev, ni, Ni, mu_i, delta_i)])

        cond = lin_dmg > 1
        # jaxprint("cond => {cond}", cond=cond)

        dmgs = lax.cond(
            cond,
            lambda x: linear_damage(x, Di, DiPrev, ni, Ni),
            lambda x: aeran_damage(x, i, Di, DiPrev, ni, Ni, mu_i, delta_i),
            lin_dmg,
        )
        # Di = dmgs["Di"]
        # DiPrev = dmgs["DiPrev"]
        Di = dmgs.at[0].get()
        DiPrev = dmgs.at[1].get()

        # if d:=nip1_tot/Ni > 1:
        #     DiPrev = Di
        #     Di = d
        # else:
        #     if i>0:
        #         inner = jnp.power(jnp.complex64(1.0-DiPrev), mu_i/delta_i)
        #         nip1_eff = (1.0-inner) * Ni
        #         nip1_tot = nip1_eff + ni

        #     inner_d = 1-jnp.power(jnp.complex64(1- nip1_tot/Ni),delta_i)
        #     DiPrev = Di
        #     Di = inner_d
    return jnp.abs(Di)
