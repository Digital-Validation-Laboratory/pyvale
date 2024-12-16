# Used for calculating 3D tensor transformation by components

import sympy as sp

r00,r11,r22,r01,r02,r12 = sp.symbols('r00 r11 r22 r01 r02 r12')
s_xx,s_yy,s_zz,s_xy,s_xz,s_yz = sp.symbols('s_xx s_yy s_zz s_xy s_xz s_yz')
d_x,d_y,d_z = sp.symbols('d_x d_y d_z')
phi = sp.symbols('phi')

r_mat = sp.Matrix([[r00,r01,r02],
                   [r01,r11,r12],
                   [r02,r12,r22]])

s_mat = sp.Matrix([[s_xx,s_xy,s_xz],
                   [s_xy,s_yy,s_yz],
                   [s_xz,s_yz,s_zz]])

d_vec = sp.Matrix([[d_x],
                   [d_y],
                   [d_z]])

s_rot = r_mat*s_mat*r_mat.T

d_rot = r_mat*d_vec

i = 2

print(80*'=')
print()
print(f'd{i}_3d=')
print(d_rot[i-1])
print()
print(80*'=')


