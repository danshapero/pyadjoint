from contextlib import ExitStack
from .optimization_solver import OptimizationSolver
from ..tape import no_annotations

try:
    from petsc4py import PETSc

    class PETScSolver(OptimizationSolver):
        r"""Use PETSc TAO to solve the given optimisation problem."""
        def __init__(self, problem, parameters=None):
            OptimizationSolver.__init__(self, problem, parameters)
            self._problem = problem

            # FIXME: Make the communicator adjustable
            comm = PETSc.COMM_SELF

            ps = [p.tape_value() for p in problem.reduced_functional.controls]
            qs = [p.copy(deepcopy=True) for p in ps]

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
            tao.setFromOptions()
            tao.setObjective(self.formObjective)
            tao.setGradient(self.formGradient, g)
            tao.setHessian(self.formHessian, H)
            tao.setSolution(x)

            self._tao = tao
            self._solution = ps
            self._work_function = qs

        @no_annotations
        def solve(self):
            self._tao.solve()
            result = self._tao.getSolution()
            result_subvecs = result.getNestSubVecs()

            ps = self._solution
            with ExitStack() as stack:
                p_subvecs = [stack.enter_context(p.dat.vec_wo) for p in ps]
                for pv, xv in zip(p_subvecs, result_subvecs):
                    pv.copy(xv)

            return self._problem.reduced_functional.controls.delist(ps)

        def formObjective(self, tao, x):
            x_subvecs = x.getNestSubVecs()
            qs = self._work_function
            with ExitStack() as stack:
                q_subvecs = [stack.enter_context(q.dat.vec_wo) for q in qs]
                for qv, xv in zip(q_subvecs, x_subvecs):
                    qv.copy(xv)

            return self._problem.reduced_functional(qs)

        def formGradient(self, tao, x, g):
            dJs = self._problem.reduced_functional.derivative()

            if isinstance(dJs, list):
                g_subvecs = g.getNestSubVecs()
                with ExitStack() as stack:
                    dJ_subvecs = [
                        stack.enter_context(dJ.dat.vec_ro) for dJ in dJs
                    ]
                    for gv, dJv in zip(g_subvecs, dJ_subvecs):
                        gv.copy(dJv)
            else:
                # FIXME this is very gross and suggests we shouldn't always
                # use nested vecs
                g_subvecs = g.getNestSubVecs()
                with dJs.dat.vec_ro as dJ:
                    g_subvecs[0].copy(dJ)

        def formObjectiveGradient(self, tao, x, g):
            pass

        def formHessian(self, tao, x, H, HP):
            pass


except ImportError:

    class PETScSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("Could not import petsc4py!")
