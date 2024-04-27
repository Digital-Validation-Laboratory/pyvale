# EXAMPLE: pymoo optimisation
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import pycave.optimisers.checkfuncs as cf

#-------------------------------------------------------------------------------
class SphereWithConstraint(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        out["G"] = 0.1 - out["F"]


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, xl=-5, xu=5, vtype=float)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)
        print(x)
        print(out["F"])

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)


class TestProblem(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, xl=-5, xu=5, vtype=float)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)


#-------------------------------------------------------------------------------
def main() -> None:
    x = cf.get_flat_x_2d((-5.12,5.12),
                        (-5.12,5.12),
                        10)
    f = cf.rastrigin(x)
    (fig,ax) = cf.plot_fun_2d('Rastrigin',
                    cf.rastrigin,
                    (-5.12,5.12),
                    (-5.12,5.12),
                    100)

    #print(f)

    return

    problem = Rastrigin()

    algorithm = PSO()

    res = minimize(problem,
                algorithm,
                seed=1,
                verbose=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    return
    (fig,ax) = cf.plot_fun_2d('Rastrigin',
                    cf.rastrigin_n,
                    (-5.12,5.12),
                    (-5.12,5.12),
                    100)

if __name__ == "__main__":
    main()