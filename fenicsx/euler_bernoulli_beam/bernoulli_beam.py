# static 1D Cantilever Beam with a rectangular cross section
# this example uses Euler-Bernoulli ("classical") beam theory
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import basix
from dolfinx.fem import (Function, functionspace, dirichletbc,
                         locate_dofs_topological,Constant,Expression)
from dolfinx.io import VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary,create_interval
from ufl import (SpatialCoordinate,inner, TestFunction, TrialFunction, div, grad, dx)
from basix.ufl import element
from dolfinx import default_scalar_type

########## GEOMETRIC INPUT ####################
E = 70e3
L = 1.0
b = 0.1
h = 0.05
rho= 2.7e-3
g = 9.81
#NOTE: floats must be converted to dolfin constants on domain below

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
NUM_ELEM = 10
domain = create_interval(MPI.COMM_WORLD, NUM_ELEM, [0, L])

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
x = SpatialCoordinate(domain)

thick = Constant(domain,h)
width = Constant(domain,b)
E = Constant(domain,E)
rho = Constant(domain,rho)
g = Constant(domain,g)

A = thick*width
EI = (E*width*thick**3)/12

#distributed load value (due to weight)
q=-rho*A*g

#################################################################
########### EXACT SOLUTIONS FOR CANTILEVER BEAM #################
#################################################################
#cantilever
w_cl = q/EI * ( (x[0]**4)/24 -(L*x[0]**3)/6 +((L**2)*x[0]**2)/4 )

#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

#define Moment expression
def M(u):
    return EI*div(grad(u))

# Create Hermite order 3 on a interval (for more informations see:
#    https://defelement.com/elements/examples/interval-Hermite-3.html )

beam_element = element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)

#finite element function space on domain, with trial and test fxns
W = functionspace(domain, beam_element)
print("Number of DOFs: %d" % W.dofmap.index_map.size_global)
print("Number of elements (intervals): %d" % NUM_ELEM)
print("Number of nodes: %d" % (NUM_ELEM+1))
u_ = TestFunction(W)
v = TrialFunction(W)

#bilinear form (LHS)
k_form = inner(div(grad(v)),M(u_))*dx

#linear form construction (RHS)
l_form = q*u_*dx

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.x.petsc_vec.localForm() as uloc:
    uloc.set(0.0)
#locate endpoints
startpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], 0))
endpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], L))

#locate DOFs from endpoints
startdof=locate_dofs_topological(W,0,startpt)
enddof=locate_dofs_topological(W,0,endpt)

#fix displacement of start point and rotation as well
fixed_disp = dirichletbc(ubc,np.array([startdof[0]]))
fixed_rot = dirichletbc(ubc,np.array([startdof[1]]))

#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
u = Function(W)

# solve variational problem
problem = LinearProblem(k_form, l_form, u=u, bcs=[fixed_disp,fixed_rot], \
                        petsc_options_prefix="linear_problem_")
uh=problem.solve()
uh.name = "Displacement and Rotation "

#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################

#save output (if uh is directly visualized in Paraview, the plot will look odd,
#  as the rotation DOFs are included in this  )
with VTKFile(domain.comm, "output/output.pvd", "w") as vtk:
    vtk.write([uh._cpp_object])
    
#NOTE: The solution uh contains both the rotation and the displacement solutions
#The rotation and displacment solutions can be separated as follows:
disp = np.empty(0)
rot = np.empty(0)
for i,x in enumerate(uh.x.array):
    if i % 2 != 0:
        rot = np.append(rot,x)
    else:
        disp = np.append(disp,x)

#evaluate derivatives and interpolate to higher order function space
T = functionspace(domain,("CG",1))

#interpolate exact ufl expression onto high-order function space
disp_expr = Expression(w_cl, T.element.interpolation_points)
disp_exact = Function(T)
disp_exact.interpolate(disp_expr)

#extract numpy arrays for plotting
exact_disp = disp_exact.x.array

x_exact = np.linspace(0,1,exact_disp.shape[0])

x_fem = np.linspace(0,1,disp.shape[0])

figure, axis = plt.subplots(1,1)

print("Maximum magnitude displacement (cantilever exact solution) is: %e" % np.min(exact_disp))
print("Maximum magnitude displacement (cantilever FEM solution) is: %e" % np.min(disp))

####PLOTTING####
#Displacement
axis.plot(x_exact,exact_disp,label='exact')
axis.plot(x_fem, disp,marker='x',linestyle=':',label='FEM')
axis.set_xlabel('x (location along beam)')
axis.set_ylabel('u(x) (transverse displacement)')
axis.set_title("Displacement of FEM vs EB theory")
axis.legend()

plt.show()