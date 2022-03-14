#!/usr/bin/env python3
import numpy
import hoomd
import itertools
import gsd.hoomd
import math
import matplotlib.pyplot as plt
import os

os.system('rm trajectory.gsd')
plt.style.use('ggplot')

def maxwell_distri(v, kT):
    return (1/2/math.pi/kT)**(3/2)*4*math.pi*v**2*numpy.exp(-v**2/2/kT)

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

# Md force object has various of per-particle loggable quantities,
# per-particle quantities have the category 'particle', e.g. {'forces': 'particle'}
print(lj.loggables)

# Add lj per-particle quantities to logger
logger = hoomd.logging.Logger()
logger.add(lj, quantities=['energies', 'forces'])

# Add manually defined velocity quantities to logger
class Status():
    def __init__(self, sim):
        self.sim = sim

    @property
    def velocity(self):
            snapshot = sim.state.get_snapshot()
            return (snapshot.particles.velocity)
status = Status(sim)
logger[('Particle', 'velocity')] = (status, 'velocity', 'particle')

# writer for generating trajectory data
gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(10000),
                             mode='xb',
                             dynamic=['property', 'momentum'],
                             filter=hoomd.filter.All(),
                             log=logger)
sim.operations.writers.append(gsd_writer)
sim.run(1e5)
# delete simulation so it is safe to open gsd files
del sim, gsd_writer, logger, integrator, nvt, lj, cell, cpu

traj = gsd.hoomd.open('trajectory.gsd', 'rb')
print(traj[0].log.keys())

# extract the velocity norm of particle and average them thorough out the whole simulation
velocity_norm = numpy.zeros(traj[0].log['particles/Particle/velocity'].shape[0])
n_frame = 0
for frame in traj:
    velocity_norm += numpy.linalg.norm(traj[0].log['particles/Particle/velocity'], ord=2, axis=1)
    n_frame += 1
velocity_norm = velocity_norm/n_frame

# calculate the maxwell-Boltzmann distribution of simulation
velocity_interval = numpy.linspace(min(velocity_norm), max(velocity_norm), 100)
velocity_dist = maxwell_distri(velocity_interval, 1.5)

# generate the hist of per-particle potential energy for last frame of calculation
fig = plt.figure(figsize=(10, 6.18))
ax = fig.add_subplot()
ax.hist(traj[-1].log['particles/md/pair/LJ/energies'], 100)
ax.set_xlabel('potential energy')
ax.set_ylabel('counts')
plt.savefig('V hist.png')


# generate the hist of velocity distribution of simulation,
# compared it with Maxwell-Boltzmann distribution
fig = plt.figure(figsize=(10, 6.18))
ax = fig.add_subplot()
ax.hist(velocity_norm, bins= 40, stacked=True, density=True, label='Simulation', edgecolor='white', linewidth=0.3)
ax.plot(velocity_interval, velocity_dist, label='Maxwell-Boltzmann distribution')
ax.legend()
ax.set_xlabel('velocity')
ax.set_ylabel('probability')
plt.savefig('velocity distribution.png')
