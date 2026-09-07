import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import sys, os
from dolfinx import plot

pwd = os.getcwd()
mesh_folder = pwd + "/meshes/"
mesh_name = "plate_with_hole"

# Initialize gmsh
gmsh.initialize()
gmsh.model.add(mesh_name)

# Parameters
L = 1.0  # Plate length
W = 0.5  # Plate width
R = 0.1  # Hole radius
lc = 0.02  # Characteristic mesh size

# Create geometry
# Points
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(L, 0, 0, lc, 2)
gmsh.model.geo.addPoint(L, W, 0, lc, 3)
gmsh.model.geo.addPoint(0, W, 0, lc, 4)

# Center of the hole
cx, cy = L/2, W/2
gmsh.model.geo.addPoint(cx, cy, 0, lc, 5)

# Points for the circular hole
gmsh.model.geo.addPoint(cx + R, cy, 0, lc, 6)
gmsh.model.geo.addPoint(cx, cy + R, 0, lc, 7)
gmsh.model.geo.addPoint(cx - R, cy, 0, lc, 8)
gmsh.model.geo.addPoint(cx, cy - R, 0, lc, 9)

# Lines for outer rectangle
gmsh.model.geo.addLine(1, 2, 1)  # Bottom
gmsh.model.geo.addLine(2, 3, 2)  # Right
gmsh.model.geo.addLine(3, 4, 3)  # Top
gmsh.model.geo.addLine(4, 1, 4)  # Left

# Circular arcs for the hole
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)

# Create curve loop and plane surface
outer_loop = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
hole_loop = gmsh.model.geo.addCurveLoop([5, 6, 7, 8])
surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

# Synchronize the model
gmsh.model.geo.synchronize()

# Create physical groups for left and right boundaries
left_boundary = gmsh.model.addPhysicalGroup(1, [4], 1)  # Tag 1 for left boundary
right_boundary = gmsh.model.addPhysicalGroup(1, [2], 2)  # Tag 2 for right boundary
domain_marker = 5
domain_surface = gmsh.model.addPhysicalGroup(2, [surface], domain_marker, name="Domain")
# Generate mesh
gmsh.model.mesh.generate(2)
# This is super important, otherwise parallel bug. Order of the finite elements
gmsh.model.mesh.setOrder(1)
        # Optimize the mesh
gmsh.model.mesh.optimize("Netgen")

# Create and import mesh to DOLFINx
mesh_data = gmshio.model_to_mesh(
    gmsh.model,
    MPI.COMM_WORLD,
    0,
    gdim=2
)

domain = mesh_data.mesh
cell_markers = mesh_data.cell_tags
facet_markers = mesh_data.facet_tags

# The mesh is now ready to use in DOLFINx
# You can access it through the 'domain' variable
print(f"Mesh has {domain.topology.index_map(2).size_global} cells and "
      f"{domain.topology.index_map(0).size_global} vertices")

facet_markers.name = "Facet markers"

if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

import pyvista
tdim = domain.topology.dim

topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.show_axes()
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)
    plotter.screenshot(mesh_folder + mesh_name + ".png")
else:
    print("Plot of the associated triangular mesh")
    plotter.show()


# Clean up gmsh
gmsh.finalize()