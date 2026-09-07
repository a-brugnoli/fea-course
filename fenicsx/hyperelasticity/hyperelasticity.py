from dolfinx import log, default_scalar_type, fem, mesh, plot
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
import pyvista
import ufl

L = 20.0
domain = mesh.create_box(
    MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [20, 5, 5], mesh.CellType.hexahedron
)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(
    domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)

d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
P = ufl.diff(psi, F)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

residual = (
    ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)
)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_stol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = NonlinearProblem(
    residual,
    u,
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="hyperelasticity",
)

# Scalar space and expression for magnitude
Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(
    ufl.sqrt(ufl.dot(u, u)), Vs.element.interpolation_points
)

# Mesh preparation
topology, cells, geometry = plot.vtk_mesh(V)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# Add initial vector and scalar data to function_grid
values = np.zeros((geometry.shape[0], 3))
values[:, : len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values

magnitude.interpolate(us)
function_grid["mag_u"] = magnitude.x.array

# Warp and explicitly set active scalars BEFORE adding to plotter
warped = function_grid.warp_by_vector("u", factor=1)
warped.set_active_scalars("mag_u")

plotter = pyvista.Plotter()
plotter.open_gif("deformation.gif", fps=3)

# Add mesh with explicitly defined scalar field
actor = plotter.add_mesh(
    warped, scalars="mag_u", show_edges=True, lighting=False, clim=[0, 10]
)

log.set_log_level(log.LogLevel.INFO)
tval0 = -1.5

for n in range(1, 10):
    T.value[2] = n * tval0
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Solver did not converge with reason {converged}."

    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")

    # 1. Update displacement array on function_grid
    function_grid["u"][:, : len(u)] = u.x.array.reshape(geometry.shape[0], len(u))

    # 2. Update magnitude function
    magnitude.interpolate(us)

    # 3. Update warped geometry coordinates and scalar data
    warped_n = function_grid.warp_by_vector("u", factor=1)
    warped.points[:, :] = warped_n.points
    warped.point_data["mag_u"][:] = magnitude.x.array

    plotter.write_frame()

plotter.close()