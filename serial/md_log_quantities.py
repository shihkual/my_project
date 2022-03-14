#!/usr/bin/env python3
import math
import numpy
import freud
import gsd.hoomd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
traj = gsd.hoomd.open('log.gsd')

# shows each loggable quantities for frame 1
traj[0].log

# call the loggable quantities individually
traj[0].log['md/compute/ThermodynamicQuantities/potential_energy']
traj[0].log['md/compute/ThermodynamicQuantities/pressure_tensor']



timestep = []
walltime = []
potential_energy = []

for frame in traj:
    timestep.append(frame.configuration.step)
    walltime.append(frame.log['Simulation/walltime'][0])
    potential_energy.append(
        frame.log['md/compute/ThermodynamicQuantities/potential_energy'][0])

fig = plt.figure(figsize=(10, 6.18))
ax = fig.add_subplot()
ax.plot(timestep, potential_energy)
ax.set_xlabel('timestep')
ax.set_ylabel('Potential Energy')
plt.savefig('V-t plot.png')

