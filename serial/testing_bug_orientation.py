#!/usr/bin/env python3
import hoomd
import gsd.hoomd
import numpy as np
import scipy.spatial
import math
import os

## statepoints ##
n_e = int(4)
n_repeats = int(1)
seed = 9487

## job documentation ##
gsd_frequency = int(3e4/10)
n_run_steps = int(3e4/10)
stop_after = int(3e4)
timestep = 0


## figure out shape vertices##
xs = np.array([np.cos(n*2*np.pi/n_e) for n in range(n_e)])
ys = np.array([np.sin(n*2*np.pi/n_e) for n in range(n_e)])
zs = np.zeros_like(ys)
vertices = np.vstack((xs, ys, zs)).T
A_particle = scipy.spatial.ConvexHull(vertices[:, :2]).volume  # in 2D, it reduce to area
vertices = vertices - np.mean(vertices, axis=0)

## initialize simulation cell ##
spacing = 3
N_particles = n_repeats ** 2
K = math.ceil(N_particles ** (1 / 2))
L = K * spacing
position = []
for i in np.linspace(-L / 2, L / 2, K, endpoint=False):
    for j in np.linspace(-L / 2, L / 2, K, endpoint=False):
        position.append((i, j, 0))
orientation = [(1, 0, 0, 0)] * N_particles

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.orientation = orientation
snapshot.particles.typeid = [0] * N_particles
snapshot.configuration.box = [L, L, 0, 0, 0, 0]
snapshot.particles.types = ['host']

## initialize simulation and integrator ##
commute = hoomd.communicator.Communicator()
cpu = hoomd.device.CPU(communicator=commute)
sim = hoomd.Simulation(device=cpu, seed=seed)

vertices_for_integrator = [tuple(list(i)) for i in vertices[:, :2]]
mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0, default_a=0)
mc.shape['host'] = dict(vertices=vertices_for_integrator)
logger = hoomd.logging.Logger()
logger.add(mc, quantities=['type_shapes'])
sim.operations.integrator = mc
sim.create_state_from_snapshot(snapshot)
sim.run(0)

## output trajectories ##
gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(gsd_frequency),
                             filter=hoomd.filter.All(),
                             mode='wb',
                             log=logger)
sim.operations.writers.append(gsd_writer)

## apply tuner to only allow rotation move of particles ##
tune_periodic = hoomd.trigger.Periodic(10, phase=0)
tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a', 'd'],
                                             target=0.33,
                                             trigger=tune_periodic,
                                             max_translation_move=0)

sim.operations.tuners.append(tune)

## run NVT MC ##
while True:
    final_timestep = stop_after
    if sim.timestep >= final_timestep:
        break
    try:
        sim.run(n_run_steps)
        timestep = sim.timestep
    except Exception as exc:
        print(exc)
        break
