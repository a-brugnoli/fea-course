import numpy as np
import sklearn.metrics.pairwise as sp

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, default_scalar_type, log, plot, io
from dolfinx.fem.petsc import (
    create_vector,
    assemble_vector,
    LinearProblem,
)
from dolfinx.mesh import (
    locate_entities_boundary,
    meshtags,
    create_rectangle,
    CellType,
)

import ufl
from pathlib import Path


def topopt(nelx, nely, volfrac, penal, rmin):

    # ============================================================
    # MPI
    # ============================================================

    comm = MPI.COMM_WORLD

    # This implementation uses a dense NumPy filter matrix.
    # Therefore it is intended to run in serial.
    if comm.size != 1:
        raise RuntimeError(
            "\nThis version of the topology optimization code "
            "uses a dense NumPy filter and must be run in serial.\n\n"
            "Use:\n"
            "    python linear_elasticity_2d.py\n\n"
            "A parallel implementation requires a distributed filter."
        )

    # ============================================================
    # Geometry and mesh
    # ============================================================

    L = 60.0
    H = 20.0

    domain = create_rectangle(
        comm,
        [
            np.array([0.0, 0.0]),
            np.array([L, H]),
        ],
        [nelx, nely],
        cell_type=CellType.quadrilateral,
    )

    # ============================================================
    # Function spaces
    # ============================================================

    # Vector-valued displacement space
    V = fem.functionspace(
        domain,
        ("Lagrange", 1, (domain.geometry.dim,))
    )

    # Piecewise constant density
    D = fem.functionspace(
        domain,
        ("DG", 0)
    )

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Test function for density space
    v_0 = ufl.TestFunction(D)

    # ============================================================
    # Density functions
    # ============================================================

    density = fem.Function(D, name="density")
    density_old = fem.Function(D, name="density_old")
    density_new = fem.Function(D, name="density_new")

    # Initial uniform density
    density.x.array[:] = volfrac

    density_old.x.array[:] = volfrac
    density_new.x.array[:] = volfrac

    if comm.rank == 0:
        print(f"Initial density: {volfrac}")

    # ============================================================
    # Boundary definitions
    # ============================================================

    def support(x):
        # Left boundary x = 0
        return np.isclose(x[0], 0.0)

    def traction(x):
        # Small section of right boundary
        return (
            np.isclose(x[0], L)
            & np.less_equal(x[1], 1.0)
        )

    fdim = domain.topology.dim - 1

    support_facets = locate_entities_boundary(
        domain,
        fdim,
        support,
    )

    traction_facets = locate_entities_boundary(
        domain,
        fdim,
        traction,
    )

    # ============================================================
    # Facet tags
    #
    # 1 = support
    # 2 = traction
    # ============================================================

    marked_facets = np.hstack(
        [
            support_facets,
            traction_facets,
        ]
    )

    marked_values = np.hstack(
        [
            np.full_like(support_facets, 1),
            np.full_like(traction_facets, 2),
        ]
    )

    sorted_facets = np.argsort(marked_facets)

    facet_tag = meshtags(
        domain,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    # ============================================================
    # Dirichlet boundary condition
    # ============================================================

    u_D = np.array(
        [0.0, 0.0],
        dtype=default_scalar_type,
    )

    support_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        support_facets,
    )

    bc = fem.dirichletbc(
        u_D,
        support_dofs,
        V,
    )

    bcs = [bc]

    # ============================================================
    # Measures
    # ============================================================

    metadata = {
        "quadrature_degree": 2
    }

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
    # Element volumes
    # ============================================================

    volume_linear = v_0 * dx
    volume_linear_form = fem.form(volume_linear)

    # In DOLFINx 0.11 create_vector expects a FunctionSpace.
    volume_vec = create_vector(D)

    assemble_vector(
        volume_vec,
        volume_linear_form,
    )

    volume_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    volume_values = volume_vec.array.copy()

    # Total physical volume
    domain_volume_local = fem.assemble_scalar(
        fem.form(1.0 * dx)
    )

    domain_volume = comm.allreduce(
        domain_volume_local,
        op=MPI.SUM,
    )

    # ============================================================
    # External load
    # ============================================================

    load = fem.Constant(
        domain,
        np.array(
            [0.0, -1.0],
            dtype=default_scalar_type,
        ),
    )

    F = ufl.dot(v, load) * ds(2)

    # ============================================================
    # Material parameters
    # ============================================================

    mu = fem.Constant(
        domain,
        default_scalar_type(0.4),
    )

    lmbda = fem.Constant(
        domain,
        default_scalar_type(0.6),
    )

    # ============================================================
    # Constitutive law
    # ============================================================

    def sigma(_u):

        epsilon = ufl.sym(
            ufl.grad(_u)
        )

        return (
            2.0 * mu * epsilon
            + lmbda
            * ufl.tr(epsilon)
            * ufl.Identity(len(_u))
        )

    # ============================================================
    # Strain energy density
    # ============================================================

    def psi(_u):

        epsilon = ufl.sym(
            ufl.grad(_u)
        )

        return (
            lmbda / 2.0
            * ufl.tr(epsilon) ** 2
            + mu
            * ufl.inner(epsilon, epsilon)
        )

    # ============================================================
    # Weak form
    # ============================================================

    K = (
        ufl.inner(
            density ** penal * sigma(u),
            ufl.grad(v),
        )
        * dx
    )

    # ============================================================
    # Linear solver
    # ============================================================

    problem = LinearProblem(
        K,
        F,
        bcs=bcs,
        petsc_options_prefix="topopt_linear_problem_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    )

    # ============================================================
    # Distance matrix for density filter
    # ============================================================

    num_elems = density.x.array.size

    # DG0 has one DOF per cell
    midpoints = D.tabulate_dof_coordinates()

    # Keep only local density DOF coordinates
    midpoints = midpoints[:num_elems]

    # Dense filter matrix
    distance_mat = np.maximum(
        rmin
        - sp.euclidean_distances(
            midpoints,
            midpoints,
        ),
        0.0,
    )

    distance_sum = distance_mat.sum(axis=1)

    if np.any(distance_sum <= 0.0):
        raise ValueError(
            "rmin is too small: some cells have no neighbours "
            "inside the filter radius."
        )

    # ============================================================
    # Results folder
    # ============================================================

    current_directory = Path(__file__).resolve().parent

    results_folder = (
        current_directory / "results"
    )

    results_folder.mkdir(
        exist_ok=True,
        parents=True,
    )

    filename = (
        results_folder / "density"
    )

    # ============================================================
    # XDMF output
    # ============================================================

    with io.XDMFFile(
        domain.comm,
        filename.with_suffix(".xdmf"),
        "w",
    ) as xdmf:

        xdmf.write_mesh(domain)

        # ========================================================
        # Optimization loop
        # ========================================================

        loop = 0
        change = 1.0

        while change > 0.01 and loop < 400:

            loop += 1

            # ----------------------------------------------------
            # Save old density
            # ----------------------------------------------------

            density_old.x.array[:] = (
                density.x.array
            )

            # ----------------------------------------------------
            # FE analysis
            # ----------------------------------------------------

            u_sol = problem.solve()

            # ----------------------------------------------------
            # Compliance
            # ----------------------------------------------------

            compliance_form = fem.form(
                density ** penal
                * psi(u_sol)
                * dx
            )

            compliance_local = fem.assemble_scalar(
                compliance_form
            )

            compliance = comm.allreduce(
                compliance_local,
                op=MPI.SUM,
            )

            # ----------------------------------------------------
            # Sensitivity
            # ----------------------------------------------------

            compliance_ufl = (
                density ** penal
                * psi(u_sol)
                * dx
            )

            dCdrho_form = fem.form(
                -ufl.derivative(
                    compliance_ufl,
                    density,
                )
            )

            # create_vector expects D, not the Form
            dCdrho_vec = create_vector(D)

            assemble_vector(
                dCdrho_vec,
                dCdrho_form,
            )

            dCdrho_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )

            # ----------------------------------------------------
            # Extract NumPy arrays
            # ----------------------------------------------------

            density_values = (
                density.x.array.copy()
            )

            dCdrho_values = (
                dCdrho_vec.array.copy()
            )

            # ----------------------------------------------------
            # Sensitivity filtering
            # ----------------------------------------------------

            filtered_sensitivity = (
                distance_mat
                @ (
                    density_values
                    * dCdrho_values
                )
            )

            dCdrho_values = (
                filtered_sensitivity
                / (
                    density_values
                    * distance_sum
                )
            )

            # ----------------------------------------------------
            # Optimality criteria update
            # ----------------------------------------------------

            l1 = 0.0
            l2 = 1e5

            move = 0.2

            while (l2 - l1) > 1e-4:

                l_mid = 0.5 * (l1 + l2)

                # Protect square root against negative
                # floating-point roundoff
                oc_argument = (
                    -dCdrho_values
                    / volume_values
                    / l_mid
                )

                oc_argument = np.maximum(
                    oc_argument,
                    1e-30,
                )

                density_new_values = np.maximum(
                    0.001,
                    np.maximum(
                        density_values - move,
                        np.minimum(
                            1.0,
                            np.minimum(
                                density_values + move,
                                density_values
                                * np.sqrt(
                                    oc_argument
                                ),
                            ),
                        ),
                    ),
                )

                density_new.x.array[:] = (
                    density_new_values
                )

                # ------------------------------------------------
                # Current volume
                # ------------------------------------------------

                current_vol_local = (
                    fem.assemble_scalar(
                        fem.form(
                            density_new * dx
                        )
                    )
                )

                current_vol = comm.allreduce(
                    current_vol_local,
                    op=MPI.SUM,
                )

                current_volfrac = (
                    current_vol
                    / domain_volume
                )

                # ------------------------------------------------
                # Update OC bounds
                # ------------------------------------------------

                if current_volfrac > volfrac:
                    l1 = l_mid
                else:
                    l2 = l_mid

            # ----------------------------------------------------
            # Convergence measure
            # ----------------------------------------------------

            change_local = np.max(
                np.abs(
                    density_new.x.array
                    - density_old.x.array
                )
            )

            change = comm.allreduce(
                change_local,
                op=MPI.MAX,
            )

            # ----------------------------------------------------
            # Volume fraction
            # ----------------------------------------------------

            current_volfrac = (
                current_vol
                / domain_volume
            )

            # ----------------------------------------------------
            # Print results
            # ----------------------------------------------------

            if comm.rank == 0:

                print(
                    "it.: {:3d}, "
                    "obj.: {:.6f}, "
                    "Vol.: {:.4f}, "
                    "ch.: {:.6f}".format(
                        loop,
                        compliance,
                        current_volfrac,
                        change,
                    )
                )

            # ----------------------------------------------------
            # Update density
            # ----------------------------------------------------

            density.x.array[:] = (
                density_new.x.array
            )

            # ----------------------------------------------------
            # Save density
            # ----------------------------------------------------

            xdmf.write_function(
                density,
                loop,
            )
# ============================================================
    # FINAL PYVISTA VISUALIZATION
    # ============================================================

    if comm.rank == 0:
        try:
            import pyvista

            print("\nCreating final PyVista visualization...")

            cells, cell_types, points = plot.vtk_mesh(domain)
            grid = pyvista.UnstructuredGrid(cells, cell_types, points)

            final_density = density.x.array.real.copy()
            grid.cell_data["density"] = final_density
            grid.set_active_scalars("density")

            # ====================================================
            # Plot 1: Continuous density
            # ====================================================

            p = pyvista.Plotter(off_screen=pyvista.OFF_SCREEN)
            p.add_mesh(grid, show_edges=True, scalar_bar_args={"title": "Density"})
            p.add_text(
                f"Topology optimization\nIteration {loop}, Volume fraction = {current_volfrac:.3f}",
                position="upper_edge",
                font_size=12,
            )
            p.view_xy()
            p.show_axes()

            density_image = results_folder / "final_density.png"

            # Pass the image path directly to show()
            p.show(screenshot=str(density_image), window_size=(1200, 500))
            print(f"Saved: {density_image}")

            # ====================================================
            # Plot 2: Thresholded final topology
            # ====================================================

            material = grid.threshold(value=0.5, scalars="density")

            p2 = pyvista.Plotter(off_screen=pyvista.OFF_SCREEN)
            p2.add_mesh(material, show_edges=False)
            p2.add_text(
                f"Final topology\nrho >= 0.5\nIteration {loop}",
                position="upper_edge",
                font_size=12,
            )
            p2.view_xy()
            p2.show_axes()

            topology_image = results_folder / "final_topology.png"

            # Pass the image path directly to show()
            p2.show(screenshot=str(topology_image), window_size=(1200, 500))
            print(f"Saved: {topology_image}")

        except ModuleNotFoundError:
            print("\nPyVista is not installed.")
            print("Install it with:\n    conda install -c conda-forge pyvista")

# ================================================================
# Main driver
# ================================================================

if __name__ == "__main__":
    n_elem = 60
    log.set_log_level(log.LogLevel.WARNING)

    topopt(
        nelx=3 * n_elem,
        nely=n_elem,
        volfrac=0.5,
        penal=3.0,
        rmin=2.0,
    )