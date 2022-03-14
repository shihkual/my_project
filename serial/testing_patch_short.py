import hoomd
import numpy as np
print("Making snapshot", flush=True)
snap = hoomd.Snapshot()
if snap.communicator.rank == 0:
    snap.particles.N = 2
    snap.particles.types = ['A']
    snap.particles.position[:] = [[0,0,0], [1.01, 0, 0]]
    snap.configuration.box = (10, 10, 10, 0, 0, 0)

device = hoomd.device.CPU()
print("Initializing", device.communicator.rank, flush=True)
sim = hoomd.Simulation(device=device, seed=1)
sim.create_state_from_snapshot(snap)

print("Parameters", device.communicator.rank, flush=True)
mc = hoomd.hpmc.integrate.Sphere()
mc.shape['A'] = dict(diameter=1)
sim.operations.integrator = mc

square_well = r'''float rsq = dot(r_ij, r_ij);
                 if (rsq < 1.21f)
                     return -10.0f;
                 else
                     return 0.0f;
'''

patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=1.1, code=square_well, param_array=[])
mc.potential = patch

gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             filter=hoomd.filter.All(),
                             mode='ab',
                             )
sim.operations.writers.append(gsd_writer)

for i in range(50):
    print("Starting run", device.communicator.rank, flush=True)
    sim.run(1000)
    snap = sim.state.get_snapshot()
    energy = patch.energy
    if snap.communicator.rank == 0:
        #print(snap.particles.position)
        #print(energy)
          print(np.linalg.norm(snap.particles.position[0,:]-snap.particles.position[1,:]))
