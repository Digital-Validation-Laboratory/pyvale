import numpy as np

c = np.array([
    [0,1,2,3],
    [4,5,6,7]
])
print(c)
print(c.shape)
print(c.flatten('F'))

n_pts = 16
n_per_e = 2

a = np.arange(0,n_pts)
print(a)
idx = np.arange(0,n_pts,n_per_e,dtype=int)
print(idx)
b = np.insert(a,idx,n_per_e)
print(b)
print()
print(np)
