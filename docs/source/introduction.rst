Introduction
============

Overview
--------

**felib** is a small educational finite element library written in Python.

The purpose of the library is to illustrate the structure and implementation
of finite element programs. The code is intentionally compact and readable so
that the numerical algorithms and software design can be understood without
the complexity typically found in production simulation software.

The library provides

- simple implementations of common finite element formulations,
- a modular framework for elements, materials, and simulations, and
- a platform suitable for experimentation, teaching, and small studies.

The emphasis of **felib** is clarity rather than performance. Many
implementation choices favor transparency of the algorithms over generality
or efficiency. The code is intended to be read and modified.


Obtaining and Installing felib
------------------------------

The recommended way to install **felib** is directly from GitHub using
``pip``:

.. code-block:: bash

   pip install git+https://github.com/tjfulle/felib.git

If you want to modify the source code or explore the implementation,
clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/tjfulle/felib.git
   cd felib
   pip install -e .

An editable installation allows changes to the source code to take effect
immediately without reinstalling the package.


Learning felib
--------------

The best way to learn how **felib** works is to explore the examples and tests
included with the library.

Example problems can be downloaded with

.. code-block:: bash

   felib fetch examples

These scripts demonstrate typical workflows and provide useful starting
points for building new simulations.

The repository also contains a collection of tests located in the
``tests/`` directory. These tests exercise individual parts of the library
and can be run using

.. code-block:: bash

   pytest

Because the tests are intentionally small and focused, they are often a
good way to understand how specific components behave.


Typical Workflow
----------------

A typical **felib** workflow consists of

1. generating or importing a mesh,
2. defining mesh regions (blocks, node sets, and side sets),
3. constructing a model and assigning elements and materials,
4. defining simulation steps and boundary conditions, and
5. running the analysis and inspecting the results.

The example below illustrates this process.


A First Example
---------------

The following script solves a small two–dimensional elasticity problem:
a plate with a circular hole subjected to traction.

The example demonstrates the core steps required to define and run a
simulation.

.. code-block:: python

   import numpy as np
   import felib


   def exercise(esize=0.05):

       # ---------------------------------------------------------------
       # Generate a mesh
       #
       # plate_with_hole returns nodal coordinates and element
       # connectivity for a triangular mesh of a plate with a hole.
       # ---------------------------------------------------------------
       nodes, elements = felib.meshing.plate_with_hole(esize=esize)
       mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)


       # ---------------------------------------------------------------
       # Define mesh regions
       #
       # Regions are used to group elements, nodes, or sides for
       # assigning materials and applying boundary conditions.
       # ---------------------------------------------------------------

       # block containing all elements
       mesh.block(
           "Block-1",
           region=lambda e: True,
           cell_type=felib.element.Tri3,
       )

       # single node near the top center used to remove rigid motion
       mesh.nodeset(
           "Point",
           region=lambda n: abs(n.x[0]) < 0.05 and n.x[1] > 0.999,
       )

       # nodes along the top edge
       mesh.nodeset(
           "Top",
           region=lambda n: n.x[1] > 0.99,
       )

       # sides along the bottom boundary
       mesh.sideset(
           "Bottom",
           region=lambda s: s.x[1] < -0.999,
       )


       # ---------------------------------------------------------------
       # Create a model and assign element and material properties
       # ---------------------------------------------------------------
       model = felib.model.Model(mesh, name="plate_with_hole")

       material = felib.material.LinearElastic(
           youngs_modulus=30e9,
           poissons_ratio=0.3,
           density=2400.0,
       )

       model.assign_properties(
           block="Block-1",
           element=felib.element.CPS3(),
           material=material,
       )


       # ---------------------------------------------------------------
       # Define and run the simulation
       # ---------------------------------------------------------------
       simulation = felib.simulation.Simulation(model)
       step = simulation.static_step()

       # fix a reference point to remove rigid body motion
       step.boundary(nodes="Point", dofs=[0, 1], value=0.0)

       # constrain vertical motion along the top boundary
       step.boundary(nodes="Top", dofs=[1], value=0.0)

       # apply downward traction on the bottom boundary
       step.traction("Bottom", magnitude=1e8, direction=[0, -1])

       # run the analysis
       simulation.run()


       # ---------------------------------------------------------------
       # Postprocess results
       # ---------------------------------------------------------------

       # nodal displacements
       u = simulation.ndata["u"]

       # displacement magnitude
       U = np.linalg.norm(u, axis=1)
       print("maximum displacement:", np.max(U))

       # visualize the deformed configuration
       scale = 0.25 / np.max(np.abs(u))
       felib.plotting.tplot(
           model.coords + scale * u,
           model.connect,
           U,
       )


   if __name__ == "__main__":
       exercise()


Library Organization
--------------------

The **felib** package is organized into several modules that correspond to
the main components of a finite element program.

``felib.mesh``  
  Mesh data structures and utilities for defining regions.

``felib.meshing``  
  Simple mesh generators used in examples and tests.

``felib.element``  
  Element formulations and element-level computations.

``felib.material``  
  Material models defining constitutive behavior.

``felib.model``  
  Model objects that combine mesh, elements, and materials.

``felib.simulation``  
  Simulation drivers, analysis steps, and solution procedures.

``felib.plotting``  
  Simple visualization utilities for inspecting results.

This modular structure mirrors the organization commonly found in larger
finite element codes while remaining small enough to be easily understood.


Degrees of Freedom
------------------

Finite element models approximate continuous fields using a finite number of
unknowns called *degrees of freedom* (DOF). These unknowns are typically
associated with nodes in the discretized domain.

For structural mechanics problems, common nodal degrees of freedom include

1. displacement in the x-direction
2. displacement in the y-direction
3. displacement in the z-direction
4. rotation about the x-axis
5. rotation about the y-axis
6. rotation about the z-axis
7. temperature

Not every analysis uses all of these degrees of freedom. For example,
two–dimensional plane stress or plane strain models typically use only the
x- and y-displacement components.

The collection of all active nodal degrees of freedom forms the global set
of unknowns solved for during the analysis.


Output
------

Simulation results produced by **felib** are written in the
`ExodusII <https://sandialabs.github.io/exodusii/>`_ file format using the
Python implementation available at

https://github.com/sandialabs/exodusii

ExodusII files can be visualized using tools such as
`ParaView <https://www.paraview.org>`_.