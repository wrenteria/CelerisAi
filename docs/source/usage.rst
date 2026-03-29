Usage
=====

.. _installation:

Installation
------------

The prerequisites for CelerisAi are standard Python libraries, with Taichi being the most important. We recommend installing in a virtual environment:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate


You can install these libraries manually:

.. code-block:: console

   $ pip install imageio>=2.36.0 matplotlib>=3.7.2 numpy>=1.24.3 scipy>=1.9.0 taichi>=1.7.0

Alternatively, you can install all required dependencies automatically using the provided requirements file:

.. code-block:: console

   $ pip install -r requirements.txt

Downloading the Source Code
-----------------------------

Clone the repository from GitHub to download the source code:

.. code-block:: console

   $ git clone https://github.com/wrenteria/CelerisAi.git


Editable Install (Recommended)
------------------------------

Install the package in editable mode so you can run examples from any folder without
modifying ``sys.path``:

.. code-block:: console

   $ pip install -e .
   

Running the Examples
--------------------

After installing the dependencies and downloading the source, you can verify the installation by running the provided examples.

For a 1D example, execute:

.. code-block:: console

   $ python setrun_1D.py

For a 2D example based on the configuration files created by CelerisWebGPU, execute:

.. code-block:: console

   $ python setrun_web.py

For more details on configuring CelerisWebGPU, please refer to its application at
`CelerisWebGPU <https://plynett.github.io/>`_.

Output cadence (logs and saved frames)
--------------------------------------

In ``Evolve``, progress logging and frame export cadence are controlled by
``plot_interval`` (default: ``100`` steps).

.. code-block:: python

   run = Evolve(solver=solver, maxsteps=10000, saveimg=True, plot_interval=25)

If ``saveimg=True``, images are saved at this same interval.

Boundary Types
--------------

For CelerisWebGPU-style cases loaded from ``config.json``, the four domain faces
use the following integer boundary types:

- ``0``: solid wall
- ``1``: sponge layer
- ``2``: incoming wave
- ``3``: periodic boundary
- ``4``: river boundary

Periodic boundaries
-------------------

Periodic boundaries are applied as opposite-side pairs. If either side of an axis
is set to ``3``, the opposite side is forced to ``3`` as well.

The current implementation also requires at least five cells along the periodic axis:

- west/east periodic requires ``nx >= 5``
- south/north periodic requires ``ny >= 5``

River boundaries
----------------

River forcing follows the WebGPU ``BoundaryPass`` type ``4`` logic and is configured
through ``config.json``.

Flood-event selection is taken from ``incident_wave_type``:

- ``10``: 10-year flood
- ``11``: 50-year flood
- ``12``: 100-year flood
- ``13``: 200-year flood
- ``14``: 500-year flood

The river hydraulics use the following parameters:

- ``river_inflow_angle``
- ``mean_upstream_channel_elevation``
- ``channel_bottom_width``
- ``channel_side_slope``
- ``channel_bank_start_upstream``
- ``channel_bank_end_upstream``
- ``Q_10``, ``Q_50``, ``Q_100``, ``Q_200``, ``Q_500``
- ``stage_10``, ``stage_50``, ``stage_100``, ``stage_200``, ``stage_500``

Current limitations in ``CelerisAi``:

- river forcing is implemented only for 2D cases
- west, east, and north boundaries are supported
- south river forcing is not implemented
- ``incident_wave_type`` must be one of ``10`` to ``14`` when a type ``4`` boundary is active

Solver Controls
---------------

Several solver options commonly used in the WebGPU-derived workflows can be passed
directly to ``Solver(...)`` or read from ``config.json`` when available.

- ``algochanges`` adjusts the wet/dry cleanup near boundaries
- ``vort_friction_factor`` enables vorticity-based momentum mixing/dissipation
- ``showBreaking`` injects breaking-derived foam/tracer into the fourth state channel
- ``clearCon`` clears that tracer/foam channel after ``Pass3`` updates when set to ``1``

For breaking-enabled visualizations, persistent foam requires:

- ``useBreakingModel=True``
- ``showBreaking=1``
- ``clearCon=0``
