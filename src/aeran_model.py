import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from jax import jit, vmap, lax, Array
from jax.debug import print as jaxprint, breakpoint
from functools import partial, reduce
import jax.numpy as jnp

from jax import config

config.update("jax_debug_nans", False)
config.update("jax_debug_infs", True)


def linear_damage(l, Di, DiPrev, ni, Ni):
    DiPrev = Di
    Di = ni / Ni
    return jnp.array([Di, DiPrev])
    # return {"Di":Di, "DiPrev":DiPrev}


def aeran_update(ni_totNi, delta_i):
    return 1 - jnp.power((1 - ni_totNi), delta_i)


# @jit
def aeran_model(n_i: Array, N_i: Array, sigma_i: Array) -> Array:
    Di = 0.0
    DiPrev = 0.0

    for i, (ni, Ni, s_i) in enumerate(zip(n_i, N_i, sigma_i)):
        lnNi = jnp.log(Ni)
        delta_i = 1.25 / lnNi
        mu_i = (sigma_i[i - 1] / s_i) ** 2.0

        lin_dmg = jnp.clip(ni / Ni, 0, 1)

        nip1_tot = ni
        if i > 0:
            exp_ = mu_i / delta_i
            inner = jnp.power(jnp.abs(1.0 - DiPrev), exp_)
            nip1_eff = (1.0 - inner) * Ni
            nip1_tot = nip1_eff + ni

        # Doesn't this defeats the purpose
        lin_dmg = jnp.clip(nip1_tot / Ni, 0, 1)
        # lin_dmg = nip1_tot / Ni

        inner_d = 1 - jnp.power((1 - lin_dmg), delta_i)
        DiPrev = Di
        Di = inner_d

    # Clip everything to ensure the
    return jnp.clip(jnp.abs(Di), 0, 1)
