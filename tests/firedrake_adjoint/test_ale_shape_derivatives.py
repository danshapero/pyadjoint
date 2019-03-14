import pytest
import numpy as np
from firedrake import *
from firedrake_adjoint import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(6, 6)


@pytest.fixture(scope='module')
def x(mesh):
    return SpatialCoordinate(mesh)


def test_sin_weak_spatial(mesh, x):
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + s)
    
    J = sin(x[0]) * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    
    V = TestFunction(S)
    dJV = div(V)*sin(x[0])*dx + V[0]*cos(x[0])*dx
    actual = assemble(dJV).vector().get_local()
    assert np.allclose(computed, actual, rtol=1e-14)


def test_tlm_assemble(mesh, x):
    tape = get_working_tape()
    tape.clear_tape()
    S =  VectorFunctionSpace(mesh, "CG", 1)
    h = Function(S)
    A = 10
    h.interpolate(as_vector((A*cos(x[1]), A*x[1])))
    f = Function(S)
    f.interpolate(as_vector((A*sin(x[1]), A*cos(x[1]))))
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)
    J = assemble(sin(x[1])* dx(domain=mesh))

    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    assert(r0 >0.95)
    Jhat(s)

    # Tangent linear model
    s.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1_tlm = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1_tlm > 1.9)
    Jhat(s)
    r1 = taylor_test(Jhat, s, h)
    assert(np.isclose(r1,r1_tlm, rtol=1e-14))


def test_shape_hessian(mesh, x):
    tape = get_working_tape()
    tape.clear_tape()
    set_working_tape(tape)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")

    mesh.coordinates.assign(mesh.coordinates + s)
    J = assemble(sin(x[1])* dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    f = Function(S, name="W")
    A = 10
    f.interpolate(as_vector((A*sin(x[1]), A*cos(x[1]))))
    h = Function(S,name="V")
    h.interpolate(as_vector((A*cos(x[1]), A*x[1])))

    # Second order taylor
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r2 > 2.9)
    Jhat(s)
    dJdmm_exact = derivative(derivative(sin(x[1])* dx(domain=mesh),x,h), x, h)
    assert(np.isclose(assemble(dJdmm_exact), Hm))


def test_PDE_hessian(mesh, x):
    tape = get_working_tape()
    tape = Tape()
    set_working_tape(tape)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)
    f = x[0]*x[1]
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    l = f*v*dx
    bc = DirichletBC(V, Constant(1), "on_boundary")
    u = Function(V)
    solve(a==l, u, bcs=bc)

    J = assemble(u*dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    f = Function(S, name="W")
    A = 10
    f.interpolate(as_vector((A*sin(x[1]), A*cos(x[1]))))
    h = Function(S,name="V")
    h.interpolate(as_vector((A*cos(x[1]), A*x[1])))

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)

    r1 = taylor_test(Jhat, s, h)
    Jhat(s)
    assert(r1>1.95)

    # First order taylor
    s.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1>1.95)
    Jhat(s)

    # # Second order taylor
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r2>2.95)
