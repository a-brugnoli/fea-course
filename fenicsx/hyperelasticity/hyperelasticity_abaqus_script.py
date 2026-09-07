# Author : Maxime Luyat

# This is a script in Abaqus to perform to same hyperelastic analysis.
# The main difference is that the simplest Neo-Hookean hyperelastic material 
# in Abaqus has the following strain energy

# \Psi = C_10 * (I_1 - 3) + 1 / D_1 * (J-1)^2

# where C_10 := G/2 and G is the shear modulus G := E/(2*(1 + \nu))
# and D_1 := 2 / K and K is the bulk modulus  K := E/(3*(1 - 2 *\nu)).

# I_1 = tr(C) where C is the Cauchy strain C = F^T F
# J = det(F) is the determinant of the deformation gradient

# In terms of Young modulus and poisson ratio the Abaqus coefficients are

# C_10 = E/(4 * (1 + \nu))
# D_1 = 6 * (1 - 2 * \nu)/E.


E = 1e4
nu = 0.3

G = E/(2*(1+nu))
K = E/(3*(1-2*nu))

C_10 = G/2
D_1 = 2/K

print(f"C_10 coefficient: {C_10}")
print(f"D_1 coefficient: {D_1}")


# -*- coding: mbcs -*-

# Module import
from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

## Model creation
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=10.0)


## Geometry
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(-0.5, -0.5), point2=(0.5, 0.5))
mdb.models['Model-1'].Part(name='HyperElasticBeam', dimensionality=THREE_D, type=DEFORMABLE_BODY)
mdb.models['Model-1'].parts['HyperElasticBeam'].BaseSolidExtrude(sketch=s, depth=20.0)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']


## Material
mdb.models['Model-1'].Material(name='HyperElasticMaterial')
mdb.models['Model-1'].materials['HyperElasticMaterial'].Hyperelastic(\
    materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, \
    volumetricResponse=VOLUMETRIC_DATA, table=((C_10, D_1), ))


## "Section" definition and assignement to geometry (Solid section for 3D part)
mdb.models['Model-1'].HomogeneousSolidSection(name='BeamSolidSection', material='HyperElasticMaterial', thickness=None)
p = mdb.models['Model-1'].parts['HyperElasticBeam']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(cells=cells)
p = mdb.models['Model-1'].parts['HyperElasticBeam']
p.SectionAssignment(region=region, sectionName='BeamSolidSection', \
                    offset=0.0, offsetType=MIDDLE_SURFACE, \
                    offsetField='', thicknessAssignment=FROM_SECTION)


## Assembly
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['HyperElasticBeam']
a.Instance(name='HyperElasticBeam-1', part=p, dependent=ON)


## Step
mdb.models['Model-1'].StaticStep(name='QuasiStaticStep', previous='Initial',\
                                maxNumInc=100000, minInc=1e-08, nlgeom=ON) #note nlgeom is ON


## Boundary conditions
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['HyperElasticBeam-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#20 ]', ), )
region = regionToolset.Region(faces=faces1)
mdb.models['Model-1'].DisplacementBC(name='Encastre', createStepName='Initial',\
            region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET, \
            amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)


## Load
e1 = a.instances['HyperElasticBeam-1'].edges
e11 = a.instances['HyperElasticBeam-1'].edges
a = mdb.models['Model-1'].rootAssembly
s1 = a.instances['HyperElasticBeam-1'].faces
side1Faces1 = s1.getSequenceFromMask(mask=('[#10 ]', ), )
region = regionToolset.Region(side1Faces=side1Faces1)
mdb.models['Model-1'].SurfaceTraction(name='Load', 
    createStepName='QuasiStaticStep', region=region, magnitude=15.0, 
    directionVector=(a.instances['HyperElasticBeam-1'].InterestingPoint(
    edge=e1[4], rule=MIDDLE), 
    a.instances['HyperElasticBeam-1'].InterestingPoint(edge=e11[10], 
    rule=MIDDLE)), distributionType=UNIFORM, field='', localCsys=None, 
    traction=GENERAL, follower=OFF, resultant=ON)


## Mesh
p = mdb.models['Model-1'].parts['HyperElasticBeam']
e = p.edges
pickedEdges = e.getSequenceFromMask(mask=('[#ed5 ]', ), )
p.seedEdgeByNumber(edges=pickedEdges, number=5, constraint=FINER)

p = mdb.models['Model-1'].parts['HyperElasticBeam']
e = p.edges
pickedEdges = e.getSequenceFromMask(mask=('[#12a ]', ), )
p.seedEdgeByNumber(edges=pickedEdges, number=20, constraint=FINER)
p = mdb.models['Model-1'].parts['HyperElasticBeam']
p.generateMesh()

a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly


## Job definition
mdb.Job(name='JobHyperElasticBeam', model='Model-1', description='', 
    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
    multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)


## Submitting
mdb.jobs['JobHyperElasticBeam'].submit(consistencyChecking=OFF)