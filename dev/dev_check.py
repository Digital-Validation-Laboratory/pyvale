import numpy as np
from functools import partial

mu, sigma = 10.0, 1.0
f = partial(np.random.default_rng().normal,loc=mu,scale=sigma)
s = f(size=10)

print(s)