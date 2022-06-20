from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
import jax.lax as lax
from jax import pmap
from jax.lax import fori_loop, cond, while_loop, scan

n=2000000
inp = jnp.arange(3, n+1, 2, dtype=jnp.int64)
aux = jnp.sqrt(n).astype(int) + 1
r = jnp.arange(2, aux, dtype=jnp.int64)

def f(r, num):
    return jnp.where(r < num, r, 2 * jnp.ones((len(r),), dtype=jnp.int64))


def prime(num):
    new_arr = f(r, num)
    pred = (jnp.mod(num, new_arr) == 0)
    return jnp.sum(pred) == 0


isprime = jit(prime, backend='cpu')

prime_filter = vmap(isprime)(inp).astype(jnp.int64)

sump = jnp.dot(inp, prime_filter)

print(sump+2)
