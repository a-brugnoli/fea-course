# Author: Jørgen Schartum Dokken
#
# This implementation is an adaptation of the work in {cite}`FenicsTutorial`
# to DOLFINx.

from mpi4py import MPI
import numpy
from pathlib import Path

from dolfinx import mesh
from dolfinx import fem
from dolfinx import io
from dolfinx import plot
from dolfinx import default_scalar_type

import ufl

# ============================================================
# 1. Create mesh
# ============================================================

domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    8,
    8,
    mesh.CellType.quadrilateral
)

# ============================================================
# 2. Function space
# ============================================================

V = fem.functionspace(domain, ("Lagrange", 1))

# ============================================================
# 3. Dirichlet boundary condition
# ============================================================

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

tdim = domain.topology.dim
fdim = tdim - 1

domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(
    V,
    fdim,
    boundary_facets
)

bc = fem.dirichletbc(uD, boundary_dofs)

# ============================================================
# 4. Variational problem
# ============================================================

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type(-6))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# ============================================================
# 5. Solve
# ============================================================

from dolfinx.fem.petsc import LinearProblem

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="linear_problem_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

uh = problem.solve()

# ============================================================
# 6. Compute error
# ============================================================

V2 = fem.functionspace(domain, ("Lagrange", 2))

uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(
    ufl.inner(uh - uex, uh - uex) * ufl.dx
)

error_local = fem.assemble_scalar(L2_error)

error_L2 = numpy.sqrt(
    domain.comm.allreduce(error_local, op=MPI.SUM)
)

error_max = numpy.max(
    numpy.abs(uD.x.array - uh.x.array)
)

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# ============================================================
# 7. Save solution
# ============================================================

current_directory = Path(__file__).resolve().parent

results_folder = current_directory / "results"
results_folder.mkdir(exist_ok=True, parents=True)

filename = results_folder / "solution"

with io.VTXWriter(
    domain.comm,
    filename.with_suffix(".bp"),
    [uh]
) as vtx:
    vtx.write(0.0)

with io.XDMFFile(
    domain.comm,
    filename.with_suffix(".xdmf"),
    "w"
) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# ============================================================
# 8. PyVista visualization
# ============================================================

try:
    import pyvista

    # --------------------------------------------------------
    # Important:
    # PyVista visualization is performed only on rank 0.
    # For this example, run the script in serial:
    #
    #     python poisson.py
    #
    # rather than:
    #
    #     mpirun -n 4 python poisson.py
    # --------------------------------------------------------

    if domain.comm.rank == 0:

        # Convert the DOLFINx function space to a VTK mesh
        cells, cell_types, points = plot.vtk_mesh(V)

        # Create PyVista grid
        grid = pyvista.UnstructuredGrid(
            cells,
            cell_types,
            points
        )

        # Add numerical solution to the mesh
        grid.point_data["u"] = uh.x.array.real

        # Make u the active scalar field
        grid.set_active_scalars("u")

        # ----------------------------------------------------
        # Create plotter
        # ----------------------------------------------------

        plotter = pyvista.Plotter()

        plotter.add_text(
            "DOLFINx Poisson solution",
            position="upper_edge",
            font_size=16
        )

        # Original 2D solution
        plotter.add_mesh(
            grid,
            show_edges=True,
            scalar_bar_args={
                "title": "u(x,y)"
            }
        )

        # ----------------------------------------------------
        # Add warped 3D surface
        # ----------------------------------------------------

        warped = grid.warp_by_scalar(
            factor=0.25
        )

        plotter.add_mesh(
            warped,
            show_edges=True,
            opacity=0.8
        )

        # Look from above / standard xy view
        plotter.view_xy()

        # ----------------------------------------------------
        # Display or save
        # ----------------------------------------------------

        if pyvista.OFF_SCREEN:
            output_image = results_folder / "solution_pyvista.png"

            plotter.screenshot(
                output_image,
                window_size=(1000, 800)
            )

            print(f"PyVista image saved to: {output_image}")

        else:
            plotter.show()

except ModuleNotFoundError:
    if domain.comm.rank == 0:
        print("PyVista is not installed.")
        print("Install it with:")
        print("    conda install -c conda-forge pyvista")