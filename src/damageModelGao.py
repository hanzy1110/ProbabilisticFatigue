import numpy as np
import matplotlib.pyplot as plt
from typing import List
from jax import jit, vmap, lax
from functools import partial
import jax.numpy as jnp

@partial(jit, static_argnums=0)
def inner(k, n_i:jnp.DeviceArray ,N_i:jnp.DeviceArray)->jnp.float32:
    if k == 0:
        return jnp.asarray((1-n_i[k]/N_i[k])**(jnp.log(N_i[k+1])/jnp.log(N_i[k])))
    else:
        return jnp.asarray((inner(k-1, n_i, N_i)-n_i[k]/N_i[k])**(jnp.log(N_i[k+1])/jnp.log(N_i[k])))

def gaoModel(n_i:jnp.DeviceArray, N_i:jnp.DeviceArray)->jnp.DeviceArray:
    totalN = len(n_i)-1
    # inner_ = vmap(inner, in_axes=(None, None, 0))

    return jnp.asarray(-1/(jnp.log(N_i[-1])) * (inner(totalN-1, n_i, N_i)-n_i[-1]/N_i[-1]))

