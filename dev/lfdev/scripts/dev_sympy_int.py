from sympy import *
init_printing(use_unicode=False)
x = Symbol('x')
y = Symbol('y')
print(integrate(integrate(x*(x-10)*y*(y-5)+20,y), x))
f_int = integrate(integrate(x*(x-10)*y*(y-5)+20,y), x)
print(f_int.subs([(x,10),(y,7.5)]).evalf()) # evaluate

