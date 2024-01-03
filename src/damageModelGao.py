import numpy as np
import matplotlib.pyplot as plt
from typing import List
from jax import jit, vmap, lax, Array
from functools import partial, reduce
import jax.numpy as jnp

@partial(jit, static_argnums=0)
def inner(k, n_i:Array ,N_i:Array, lnN_i)->jnp.float32:

    if k == 0:
        exponent = jnp.true_divide(lnN_i[k+1],lnN_i[k])
        # nNi = jnp.minimum(jnp.true_divide(n_i[k]-1,N_i[k]), 1)
        nNi = jnp.true_divide(n_i[k]-1,N_i[k])
        return jnp.asarray(jnp.power(1-nNi,exponent))

    else:
        exponent = jnp.true_divide(lnN_i[k+1],lnN_i[k])
        nNi = jnp.true_divide(n_i[k]-1,N_i[k])
        return jnp.asarray(jnp.power(inner(k-1, n_i, N_i, lnN_i)-nNi,exponent))

# @profile
def gaoModel(n_i:Array, N_i:Array, lnN_i)->Array:
    totalN = len(n_i)-1
    # inner_ = vmap(inner, in_axes=(None, None, 0))
    nNi = jnp.true_divide(n_i[-1],N_i[-1])
    return jnp.asarray(-1/(lnN_i[-1]) * jnp.log(inner(totalN-1, n_i, N_i, lnN_i)-nNi))

def inner_debug(k, n_i:Array ,N_i:Array, lnN_i)->jnp.float32:

    if k == 0:
        # exponent = np.true_divide(np.log(N_i[k+1]),np.log(N_i[k]))
        exponent = np.true_divide(lnN_i[k+1],lnN_i[k])
        nNi = np.true_divide(n_i[k]-1,N_i[k])
        print('nNi->', nNi)
        print('exponent->', exponent)
        return np.power(1-nNi,exponent)

    else:
        # exponent = np.true_divide( np.log(N_i[k+1]),np.log(N_i[k]))
        exponent = np.true_divide(lnN_i[k+1],lnN_i[k])
        nNi = np.true_divide(n_i[k],N_i[k])
        print('nNi->', nNi)
        print('exponent->', exponent)
        return np.power(inner_debug(k-1, n_i, N_i, lnN_i)-nNi,exponent)

def gaoModel_debug(n_i:Array, N_i:Array, lnN_i)->Array:
    totalN = len(n_i)-1
    # inner_ = vmap(inner, in_axes=(None, None, 0))

    nNi = n_i[-1]/N_i[-1]
    print('outer nNi', nNi)
    return np.asarray(-1/(lnN_i[-1]) * np.log(inner_debug(totalN-1, n_i, N_i, lnN_i)-nNi))

def minerRule(n_i:Array, N_i:Array,)->Array:
    # n_iS = n_i[n_i<N_i]
    # N_iS = N_i[n_i<N_i]
    return jnp.true_divide(n_i, N_i).sum()

