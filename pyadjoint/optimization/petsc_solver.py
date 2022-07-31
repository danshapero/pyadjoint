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

        def __init__(self, problem, parameters):
            OptimizationSolver.__init__(self, problem, parameters)
            self._petsc_objective = PETScObjective(problem.reduced_functional)

            # FIXME: Make the communicator adjustable
            comm = PETSc.COMM_SELF

            ps = [p.tape_value() for p in problem.reduced_functional.controls]
            with ExitStack() as stack:
                # FIXME: This is specific to Firedrake probably
                vecs = [stack.enter_context(p.dat.vec_ro) for in ps]
                x = PETSc.Vec().createNest(vecs, comm=comm)

            H = PETSc.Mat().create(comm)
            H.setSizes([size, size])
            H.setFromOptions()
            H.setOption(PETSc.Mat.Option.Symmetric, True)
            H.setUp()

            tao = PETSc.TAO().create(comm)
            tao.setObjective(self._petsc_objective.formObjective)
            tao.setGradient(self._petsc_objective.formGradient)
            tao.setHessian(self._petsc_objective.formHessian, H)
            tao.setSolution(x)

        @no_annotations
        def solve(self):
            raise NotImplementedError("Not quite baked yet sorry bruv")

except ImportError:

    class PETScSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("Could not import petsc4py!")
