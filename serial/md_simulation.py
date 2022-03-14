#!/usr/bin/env python3
import numpy
import hoomd
import itertools
import gsd.hoomd
import math

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu)
sim.create_state_from_gsd('random.gsd')

integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell()
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)
sim.operations.integrator = integrator
sim.run(0)

# check for loggable quantities, which are the class properties or method
sim.loggables
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
thermodynamic_properties.loggables

# create logger
logger = hoomd.logging.Logger()
#You can add loggable quantities from any number of objects to a Logger. Logger uses the namespace of the class to assign a unique name for each quantity.
logger.add(thermodynamic_properties)  # use all the quantities provide by ther_prop
logger.add(sim, quantities=['timestep', 'walltime'])  #Or, I can specify the quantities

gsd_writer = hoomd.write.GSD(filename='log.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             mode='wb',
                             filter=hoomd.filter.Null(), # use this method to only store the quantities we specify for logger while ignore the particle data. This saves the file size and run time.
			     log=logger) 
sim.operations.writers.append(gsd_writer)
sim.run(1e5)

