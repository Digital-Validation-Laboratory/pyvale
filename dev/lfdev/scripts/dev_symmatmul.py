# Used for calculating 3D tensor transformation by components

import sympy as sp

r00,r11,r22,r01,r02,r12 = sp.symbols('r00 r11 r22 r01 r02 r12')
s_xx,s_yy,s_zz,s_xy,s_xz,s_yz = sp.symbols('s_xx s_yy s_zz s_xy s_xz s_yz')
phi = sp.symbols('phi')

r_mat = sp.Matrix([[r00,r01,r02],
                   [r01,r11,r12],
                   [r02,r12,r22]])

s_mat = sp.Matrix([[s_xx,s_xy,s_xz],
                   [s_xy,s_yy,s_yz],
                   [s_xz,s_yz,s_zz]])

s_rot = r_mat*s_mat*r_mat.T

i = 1
j = 2

print(80*'=')
print()
print(f's{i}{j}_3d=')
print(s_rot[i-1,j-1])
print()
print(80*'=')

r_2d = sp.Matrix([[r00,r01],
                   [r01,r11]])

s_2d = sp.Matrix([[s_xx,s_xy],
                   [s_xy,s_yy]])

s_r_2d = r_2d*s_2d*r_2d.T
print(80*'=')
print()
print(f's{i}{j}_2d=')
print(s_r_2d[i-1,j-1])
print()
print(80*'=')
