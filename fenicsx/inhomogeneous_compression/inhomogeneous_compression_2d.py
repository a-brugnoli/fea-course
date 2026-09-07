from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista
import ufl

from mpi4py import MPI

from dolfinx import fem, geometry, log, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem


# ============================================================
# Mesh and finite element space
# ============================================================

L = 10.0
n_elements = 30
pol_degree = 2

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [L, L]],
    [n_elements, n_elements],
    cell_type=mesh.CellType.quadrilateral,
)

gdim = domain.geometry.dim

V = fem.functionspace(
    domain,
    ("Lagrange", pol_degree, (gdim,)),
)


# ============================================================
# Boundary markers
# ============================================================

def left(x):
    return np.isclose(x[0], 0.0)


def right(x):
    return np.isclose(x[0], L)


def bottom(x):
    return np.isclose(x[1], 0.0)


def top(x):
    return np.isclose(x[1], L)


def load_location(x):
    return np.logical_and(
        np.isclose(x[1], L),
        x[0] <= L / 2,
    )


fdim = domain.topology.dim - 1

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

load_facets = mesh.locate_entities_boundary(domain, fdim, load_location)


# Tag loaded facets with marker 5
marked_facets = np.asarray(load_facets, dtype=np.int32)
marked_values = np.full(marked_facets.shape, 5, dtype=np.int32)

sort_order = np.argsort(marked_facets)

facet_tag = mesh.meshtags(
    domain,
    fdim,
    marked_facets[sort_order],
    marked_values[sort_order],
)


# ============================================================
# Dirichlet boundary conditions
# ============================================================

left_dofs_x = fem.locate_dofs_topological(
    V.sub(0), fdim, left_facets
)

top_dofs_x = fem.locate_dofs_topological(
    V.sub(0), fdim, top_facets
)

bottom_dofs_y = fem.locate_dofs_topological(
    V.sub(1), fdim, bottom_facets
)

u_bc = default_scalar_type(0)

bcs = [
    fem.dirichletbc(u_bc, left_dofs_x, V.sub(0)),
    fem.dirichletbc(u_bc, top_dofs_x, V.sub(0)),
    fem.dirichletbc(u_bc, bottom_dofs_y, V.sub(1)),
]


# ============================================================
# Nonlinear hyperelastic problem
# ============================================================

traction = fem.Constant(
    domain,
    default_scalar_type((0.0, 0.0)),
)

v = ufl.TestFunction(V)
u = fem.Function(V)

# Spatial dimension
d = gdim

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# ============================================================
# Material parameters
# ============================================================

mu = fem.Constant(domain, default_scalar_type(80.194))       # N/mm^2
lmbda = fem.Constant(domain, default_scalar_type(400889.8))  # N/mm^2


# Compressible neo-Hookean strain energy
psi = (
    (mu / 2.0) * (Ic - 3.0)
    - mu * ufl.ln(J)
    + (lmbda / 2.0) * ufl.ln(J) ** 2
)

# First Piola-Kirchhoff stress
P = ufl.diff(psi, F)


# ============================================================
# Integration measures
# ============================================================

metadata = {"quadrature_degree": 4}

dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata=metadata,
)

ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_tag,
    metadata=metadata,
)


# ============================================================
# Residual and Jacobian
# ============================================================

residual = (
    ufl.inner(ufl.grad(v), P) * dx
    - ufl.inner(v, traction) * ds(5)
)

du = ufl.TrialFunction(V)

jacobian = ufl.derivative(
    residual,
    u,
    du,
)


# ============================================================
# PETSc SNES nonlinear solver
# ============================================================
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",

    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "snes_stol": 1.0e-8,
    "snes_max_it": 50,

    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",

    "snes_error_if_not_converged": True,
}

problem = NonlinearProblem(
    residual,
    u,
    bcs=bcs,
    J=jacobian,
    petsc_options_prefix="hyperelasticity_",
    petsc_options=petsc_options,
)

solver = problem.solver


# ============================================================
# Output directories
# ============================================================

current_directory = Path(__file__).resolve().parent

results_folder = current_directory / "results"
results_folder.mkdir(exist_ok=True, parents=True)

comm = domain.comm
rank = comm.rank
size = comm.size


# ============================================================
# PyVista visualization
# ============================================================
#
# The mechanics solve itself is MPI-compatible. The PyVista GIF
# section below is kept for serial execution because vtk_mesh(u.V)
# is process-local.

visualize = size == 1

if visualize:
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter()

    plotter.open_gif(
        str(results_folder / "displacement.gif"),
        fps=10,
    )

    topology, cells, geometry_coords = plot.vtk_mesh(
        u.function_space
    )

    function_grid = pyvista.UnstructuredGrid(
        topology,
        cells,
        geometry_coords,
    )

    values = np.zeros(
        (geometry_coords.shape[0], 3),
        dtype=u.x.array.dtype,
    )

    values[:, :gdim] = u.x.array.reshape(
        geometry_coords.shape[0],
        gdim,
    )

    function_grid["u"] = values
    function_grid.set_active_vectors("u")

    # Warp mesh by displacement
    warped = function_grid.warp_by_vector(
        "u",
        factor=1.0,
    )

    warped.set_active_vectors("u")

    plotter.add_mesh(
        warped,
        show_edges=True,
        lighting=False,
        clim=[0, 10],
    )

    plotter.show_axes()
    plotter.view_xy()


# ============================================================
# Displacement magnitude
# ============================================================

Vs = fem.functionspace(
    domain,
    ("Lagrange", pol_degree),
)

magnitude = fem.Function(Vs)

magnitude_expr = fem.Expression(
    ufl.sqrt(
        sum(u[i] ** 2 for i in range(gdim))
    ),
    Vs.element.interpolation_points,
)

magnitude.interpolate(magnitude_expr)


if visualize:
    warped["mag"] = magnitude.x.array
    warped.set_active_scalars("mag")


# ============================================================
# Point at which displacement is monitored
# ============================================================

top_point = np.array(
    [[0.0, L, 0.0]],
    dtype=np.float64,
)

# Find cell(s) containing the point
bb_tree = geometry.bb_tree(
    domain,
    domain.topology.dim,
)

cell_candidates = geometry.compute_collisions_points(
    bb_tree,
    top_point,
)

colliding_cells = geometry.compute_colliding_cells(
    domain,
    cell_candidates,
    top_point,
)

local_cells = colliding_cells.links(0)

if len(local_cells) > 0:
    point_cell = np.int32(local_cells[0])
else:
    point_cell = np.int32(-1)


# ============================================================
# Load stepping
# ============================================================

log.set_log_level(log.LogLevel.INFO)

tval_fin = -600.0

n_times = 150
tval0 = tval_fin / n_times

load_time = np.linspace(
    0.0,
    tval_fin,
    n_times + 1,
)

u_y_point = np.zeros(
    n_times + 1,
)


# ============================================================
# Nonlinear solve
# ============================================================
for n in range(1, n_times + 1):

    traction.value[1] = n * tval0

    try:
        problem.solve()
    except Exception as exc:
        if rank == 0:
            print(f"\nSolver exception at load step {n}")
            print(exc)
        raise

    reason = solver.getConvergedReason()
    num_its = solver.getIterationNumber()

    if rank == 0:
        print(
            f"Step {n:3d}: "
            f"load = {traction.value[1]: .6f}, "
            f"iterations = {num_its}, "
            #f"SNES reason = {reason}"
        )

    if reason < 0:
        raise RuntimeError(
            f"SNES diverged at load step {n}, "
            f"reason = {reason}"
        )

    u.x.scatter_forward()

    # Point evaluation
    local_uy = 0.0

    if point_cell >= 0:
        value = u.eval(
            top_point[0],
            np.array([point_cell], dtype=np.int32),
        )

        local_uy = float(value[1])

    u_y_point[n] = comm.allreduce(
        local_uy,
        op=MPI.SUM,
    )

    if visualize:
        function_grid["u"][:, :gdim] = u.x.array.reshape(
            geometry_coords.shape[0],
            gdim,
        )

        magnitude.interpolate(magnitude_expr)

        warped_new = function_grid.warp_by_vector(
            "u",
            factor=1.0,
        )

        warped.points[:, :] = warped_new.points
        warped.point_data["mag"][:] = magnitude.x.array

        plotter.update_scalar_bar_range([0, 10])
        plotter.write_frame()
# ============================================================
# Close visualization
# ============================================================

if visualize:
    plotter.close()


# ============================================================
# Load-displacement curve
# ============================================================

if rank == 0:

    plt.figure()

    plt.plot(
        -u_y_point,
        -load_time,
        "+",
    )

    plt.xlabel(
        "Displacement at top left point (mm)"
    )

    plt.ylabel(
        "Applied traction (N/mm²)"
    )

    plt.title(
        "Load-Displacement Curve at Top Left Point"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        results_folder / "load_displacement_curve.pdf",
        format="pdf",
    )

    plt.close()