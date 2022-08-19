from contextlib import ExitStack
from .optimization_solver import OptimizationSolver
from ..tape import no_annotations

try:
    from petsc4py import PETSc

    class PETScObjective:
        def __init__(self, rf):
            self.rf = rf

        def formObjective(self, tao, x):
            pass

        def formGradient(self, tao, x, g):
            pass

        def formObjectiveGradient(self, tao, x, g):
            pass

        def formHessian(self, tao, x, H, HP):
            pass


    class PETScSolver(OptimizationSolver):
        r"""Use PETSc TAO to solve the given optimisation problem."""

        def __init__(self, problem, parameters=None):
            OptimizationSolver.__init__(self, problem, parameters)
            self._petsc_objective = PETScObjective(problem.reduced_functional)
            self._problem = problem

            # FIXME: Make the communicator adjustable
            comm = PETSc.COMM_SELF

            ps = [p.tape_value() for p in problem.reduced_functional.controls]
            with ExitStack() as stack:
                # FIXME: This is specific to Firedrake probably
                vecs = [stack.enter_context(p.dat.vec_ro) for p in ps]
                x = PETSc.Vec().createNest(vecs, comm=comm)

            g = x.duplicate()

            size = x.getSize()
            H = PETSc.Mat().create(comm)
            H.setSizes([size, size])
            H.setFromOptions()
            H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            H.setUp()

            tao = PETSc.TAO().create(comm)
            tao.setObjective(self._petsc_objective.formObjective)
            tao.setGradient(self._petsc_objective.formGradient, g)
            tao.setHessian(self._petsc_objective.formHessian, H)
            tao.setSolution(x)

            self._tao = tao
            self._solution = ps

        @no_annotations
        def solve(self):
            self._tao.solve()
            result = self._tao.getSolution()
            result_subvecs = result.getNestSubVecs()

            ps = self._solution
            with ExitStack() as stack:
                solution_subvecs = [stack.enter_context(p.dat.vec_wo) for p in ps]
                for pv, xv in zip(solution_subvecs, result_subvecs):
                    pv.copy(xv)

            return self._problem.reduced_functional.controls.delist(ps)

except ImportError:

    class PETScSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("Could not import petsc4py!")
