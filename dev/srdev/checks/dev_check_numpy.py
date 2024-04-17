import numpy as np
from pprint import pprint

a = np.random.default_rng().uniform(low=-1.0,high=1.0,size=(10,1))
pprint(a)
t = np.tile(a,(1,3))
print(t.shape)

