#!/usr/bin/env python3
import hoomd
import gsd.hoomd
import numpy as np
import itertools
import scipy.spatial
import math
import os
import patch_c_code


## state points ##
n_e = int(3)                          # n_edges
f = 1                                  # patch_offset, 0 = middle, ±1 = on vertex
n_repeats = int(3)               # n_repeats
initial_config = 'dilute'   # initial_config
seed = 9487                            # replica
kT = 0.4                                 # kT
pressure = 1.5                      # pressure
use_floppy_box = True
epsilon_ratio = 2
sp_phi = 0.65
sigma = 1
lambdasigma = 0.1
repulsive_radius = 0.6


## job documentation ##
doc_mc_d = None                             # MC rotation trial move
doc_mc_a = None                             # MC displacement trial move
doc_vertices = None                      # vertices of particles
doc_A_particle = None                  # Area of particles
doc_gsd_frequency = int(1e5/2)
doc_thermo_frequency = int(1e5/2)
doc_n_tune_blocks = 20
doc_n_tune_steps = 1000
doc_n_run_blocks = 1000
doc_n_run_steps = int(3e5)
doc_scale = 0.01                                  # compress scale
doc_do_tuning = True
doc_compressed = False
doc_volume_delta = 0.001                    # job.doc.get
doc_shear_delta = (0.001, 0, 0)         # job.doc.get
doc_aspect_delta = 0.001                    # job.doc.get
doc_length_delta = (0.01, 0.01, 0)    # job.doc.get
doc_stop_after = int(50e6)
doc_continue_running = True
doc_timestep = 0



class Walltimereach(hoomd.custom.Action):

    def __init__(self, sim):
        self.sim = sim

    def act(self, timestep):
        buffer_time = 60*30
        if self.sim.walltime >= 60*60*7 - buffer_time:
            raise Exception(f"Wall time reached!! \n"
                            f"compressed timesteps: {compressing_timestep}\n"
                            f"total timestep: {timestep}\n"
                            f"elapsed time (s): {self.sim.walltime}\n"
                            f"averaged tps (s^-1): {timestep/self.sim.walltime}")
        else:
            return

# help calling the quantities for individual logger
class Thermo_status():
    def __init__(self,sim):
        self.sim = sim

    @property
    def volume(self):
        return self.sim.state.box.volume

    @property
    def lx(self):
        return self.sim.state.box.Lx

    @property
    def ly(self):
        return self.sim.state.box.Ly

    @property
    def lz(self):
        return self.sim.state.box.Lz

    @property
    def xy(self):
        return self.sim.state.box.xy

class MC_status():
    def __init__(self, mc):
        self.mc = mc

    @property
    def trans_accep(self):
        total_moves = sum(self.mc.translate_moves)
        accept_moves = self.mc.translate_moves[0]
        if total_moves == 0:
            return 0
        else:
            return accept_moves / total_moves

    @property
    def rotat_accep(self):
        total_moves = sum(self.mc.rotate_moves)
        accept_moves = self.mc.rotate_moves[0]
        if total_moves == 0:
            return 0
        else:
            return accept_moves / total_moves

    @property
    def d_trial(self):
        return self.mc.d.default

    @property
    def a_trial(self):
        return self.mc.a.default

    @property
    def sweep(self):
        return

class Compress_status():
    def __init__(self, sim, mc):
        self.sim = sim
        self.mc = mc
        snapshot = sim.state.get_snapshot()
        N_types = {ptype: 0 for ptype in snapshot.particles.types}
        types = 0
        for p in snapshot.particles.types:
            n_temp = sum(snapshot.particles.typeid == types)
            N_types[p] = n_temp
        A_particles = 0
        for ptype, count in N_types.items():
            A_particles += doc_A_particle * count
        self.a_particles = A_particles
        snapshot.communicator.barrier()

    @property
    def density(self):
        return self.a_particles / self.sim.state.box.volume

    @property
    def count_overlaps(self):
        return self.mc.overlaps

class Boxmc_status():
    def __init__(self, box_mc):
        self.box_mc = box_mc

    @property
    def volume_acceptance(self):
        total_moves = sum(self.box_mc.volume_moves)
        accept_moves = self.box_mc.volume_moves[0]
        if total_moves == 0:
            return 0
        else:
            return accept_moves / total_moves

    @property
    def shear_acceptance(self):
        total_moves = sum(self.box_mc.shear_moves)
        accept_moves = self.box_mc.shear_moves[0]
        if total_moves == 0:
            return 0
        else:
            return accept_moves / total_moves

    @property
    def aspect_acceptance(self):
        total_moves = sum(self.box_mc.aspect_moves)
        accept_moves = self.box_mc.aspect_moves[0]
        if total_moves == 0:
            return 0
        else:
            return accept_moves / total_moves

    @property
    def box_betaP(self):
        return box_mc.betaP.value

def generate_patch_location_c_code(patch_locations):
    ret_str = ''
    for pl in patch_locations:
        ret_str += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
    return ret_str

def done_running():
    if doc_n_run_blocks is None:
        return False
    tts = doc_n_run_steps * doc_n_run_blocks
    end_time = tts # job.doc.get('stop_after', tts)
    cr = doc_continue_running # job.doc.get('continue_running', True)
    ts_criterion = doc_timestep > end_time # ts_criterion = job.doc.get('timestep', 0) > end_time
    stopping_criteria = (not cr, ts_criterion)
    return any(stopping_criteria)

def timestep_label():
    if done_running():
        return 'done running'
    ts = doc_timestep  #job.doc.get('timestep', 0)
    if ts is None:
        return False
    ts_str = np.format_float_scientific(ts, precision=1,
            sign=False, exp_digits=1).replace('+', '')
    sa_str = np.format_float_scientific(doc_stop_after, precision=1,
            sign=False, exp_digits=1).replace('+', '')
    return 'step: {} -> {}'.format(ts_str, sa_str)



## figure out shape vertices and patch location ##
xs = np.array([np.cos(n*2*np.pi/n_e) for n in range(n_e)])
ys = np.array([np.sin(n*2*np.pi/n_e) for n in range(n_e)])
zs = np.zeros_like(ys)
vertices = np.vstack((xs, ys, zs)).T
A_particle = scipy.spatial.ConvexHull(vertices[:, :2]).volume  # in 2D, it reduce to area
vertices = vertices - np.mean(vertices, axis=0)
vertex_vertex_vectors = np.roll(vertices, -1, axis=0) - vertices
half_edge_locations = vertices + 0.5 * vertex_vertex_vectors
patch_locations = half_edge_locations + f * (vertices - half_edge_locations)


## build simualtion cell ##
if initial_config == 'dilute':
    spacing = 2
    N_particles = n_repeats**2
    K = math.ceil(N_particles**(1 / 2))
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
    with gsd.hoomd.open(name='restart.gsd', mode='wb') as f:
        f.append(snapshot)
else:
    raise NotImplementedError('Initialization not implemented.')

## set up integrator ##
commute = hoomd.communicator.Communicator()
cpu = hoomd.device.CPU(communicator=commute)
sim = hoomd.Simulation(device=cpu, seed=seed)

vertices_for_integrator = [tuple(list(i)) for i in vertices[:, :2]]
mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0, default_a=0)
mc.shape['host'] = dict(vertices=vertices_for_integrator)
logger = hoomd.logging.Logger()
logger.add(mc, quantities=['type_shapes'])
sim.operations.integrator = mc

sim.create_state_from_gsd('restart.gsd')
sim.run(1)
mc.a.default = 0.5
mc.d.default = 0.1

if commute.num_ranks == 1:
    snapshot = sim.state.get_snapshot()
    doc_mc_d = {x: 0.1 for x in snapshot.particles.types}
    doc_mc_a = {x: 0.1 for x in snapshot.particles.types}
    doc_vertices = vertices
    doc_patch_locations = patch_locations
    doc_A_particle = A_particle
    os.system('cp {} {}'.format('restart.gsd', 'init.gsd'))
commute.barrier()
hoomd.write.GSD.write(state=sim.state, mode='wb', filename='restart.gsd', log=logger)


## Sampling ##
if pressure != None:
    betaP = pressure / kT
else:
    betaP = None
seed = seed
use_floppy_box = use_floppy_box
epsilon_ratio = epsilon_ratio
gsd_frequency = doc_gsd_frequency
thermo_frequency = doc_thermo_frequency
n_tune_blocks = doc_n_tune_blocks
n_tune_steps = doc_n_tune_steps
n_run_blocks = doc_n_run_blocks
n_run_steps = doc_n_run_steps
mc_d, mc_a = doc_mc_d, doc_mc_a
vertices = doc_vertices[:, :2]
scale = doc_scale
do_tuning = doc_do_tuning
volume_delta = doc_volume_delta
shear_delta = doc_shear_delta
aspect_delta = doc_aspect_delta
length_delta = doc_length_delta

# handle hoomd message files: save old output to a file
msg_fn = os.path.join(os.path.curdir, 'hoomd-log.txt')  # need to use job.fn('hoomd-log.txt')
if os.path.isfile(msg_fn):
    if commute.num_ranks == 1:
        with open(os.path.join(os.path.curdir, 'previous-hoomd-logs.txt'), 'a') as outfile:
            with open(msg_fn, 'r') as infile:
                for line in infile:
                    outfile.write(line)

commute = hoomd.communicator.Communicator()
cpu = hoomd.device.CPU(communicator=commute, msg_file=msg_fn)
sim = hoomd.Simulation(device=cpu, seed=seed)
mc = hoomd.hpmc.integrate.ConvexPolygon()
mc.shape['host'] = dict(vertices=vertices)
sim.operations.integrator = mc
logger = hoomd.logging.Logger()
logger.add(mc, quantities=['type_shapes'])

commute.barrier()

# initialize system from restart file
initial_gsd_fn = os.path.join(os.path.curdir, 'init.gsd')  # need to use job.fn()
restart_fn = os.path.join(os.path.curdir, 'restart.gsd')  # need to use job.fn()
sim.create_state_from_gsd(filename=restart_fn)

gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(gsd_frequency),
                             filter=hoomd.filter.All(),
                             mode='ab',
                             log=logger)
sim.operations.writers.append(gsd_writer)
restart_frequency = gsd_frequency
restart_writer = hoomd.write.GSD(filename=restart_fn,
                             trigger=hoomd.trigger.Periodic(gsd_frequency),
                             filter=hoomd.filter.All(),
                             truncate=True,
                             mode='ab',
                             log=logger)
sim.operations.writers.append(restart_writer)

tune_periodic = hoomd.trigger.Periodic(10, phase=0)
tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a', 'd'],
                                             target=0.33,
                                             trigger=tune_periodic,)

sim.operations.tuners.append(tune)

sim.run(0)

gsd_writer.write(state=sim.state,
                              mode='wb',
                              filename='trajectory.gsd',
                              log=logger)
restart_writer.write(state=sim.state,
                                     mode='wb',
                                     filename=restart_fn,
                                     log=logger)

# create loggers
thermo_logger = hoomd.logging.Logger(categories=['scalar'])
thermo_status = Thermo_status(sim)
thermo_logger[('volume')] = (thermo_status, 'volume', 'scalar')
thermo_logger[('lx')] = (thermo_status, 'lx', 'scalar')
thermo_logger[('ly')] = (thermo_status, 'ly', 'scalar')
thermo_logger[('lz')] = (thermo_status, 'lz', 'scalar')
thermo_logger[('xy')] = (thermo_status, 'xy', 'scalar')

thermo_writer = hoomd.write.Table(output=open('thermo.txt', mode='w', newline='\n'),
                               trigger=hoomd.trigger.Periodic(thermo_frequency),
                               logger=thermo_logger)
sim.operations.writers.append(thermo_writer)

mc_logger = hoomd.logging.Logger(categories=['scalar'])
mc_status = MC_status(mc)
mc_logger[('trans_accep')] = (mc_status, 'trans_accep', 'scalar')
mc_logger[('rotat_accep')] = (mc_status, 'rotat_accep', 'scalar')

mc_writer = hoomd.write.Table(output=open('hpmc.txt', mode='w', newline='\n'),
                               trigger=hoomd.trigger.Periodic(thermo_frequency),
                               logger=mc_logger)
sim.operations.writers.append(mc_writer)

compress_logger = hoomd.logging.Logger(categories=['scalar', 'string'])
compress_status = Compress_status(sim, mc)
compress_logger.add(sim, quantities=['timestep'])
compress_logger[('density')] = (compress_status, 'density', 'scalar')
compress_logger[('n_overlaps')] = (compress_status, 'count_overlaps', 'string')

compress_writer = hoomd.write.Table(output=open('compress.txt', mode='w', newline='\n'),
                               trigger=hoomd.trigger.Periodic(1000),
                               logger=compress_logger)
sim.operations.writers.append(compress_writer)

# determine the target box volume
snapshot = sim.state.get_snapshot()
N_types = {ptype: 0 for ptype in snapshot.particles.types}
types = 0
for p in snapshot.particles.types:
    n_temp = sum(snapshot.particles.typeid == types)
    N_types[p] = n_temp
A_particles = 0
for ptype, count in N_types.items():
    A_particles += doc_A_particle * count
A_target = A_particles / sp_phi
initial_box = sim.state.box
phi = A_particles / initial_box.volume
final_box = hoomd.Box.from_box(initial_box)
final_box.volume = A_target
sim.run(1)

# add compressor
n_expand_steps = 0
need_to_compress = not doc_compressed
compress = hoomd.hpmc.update.QuickCompress(
    trigger=hoomd.trigger.Periodic(100),
    target_box=final_box,
    min_scale=scale
)
sim.operations.updaters.append(compress)

# compressing the box
while not compress.complete and sim.timestep < 1e6 and need_to_compress:
    sim.run(1000)
if not compress.complete:
    # return
    doc_trouble_compressing = True  # job.doc['trouble_compressing'] = True
    doc_compressed = False
    commute.barrier()
    raise RuntimeError("Compression failed to complete")
else:
    compressing_timestep = sim.timestep  # job.doc.setdefault('compressed_timestep', hoomd.get_step())
    doc_compressed = True
    commute.barrier()

del compress, compress_logger, compress_writer

# box moves
box_moves = None
if betaP != None:
    box_mc = hoomd.hpmc.update.BoxMC(betaP, trigger=hoomd.trigger.Periodic(10))
    box_mc.volume = {'mode': 'ln', 'weight': 1.0, 'delta': volume_delta}
    log_quantities = ['volume_acceptance', 'box_betaP']

# add floppy box moves if needed
if use_floppy_box:
    if box_moves == None:
         box_mc = hoomd.hpmc.update.BoxMC(betaP)
    box_mc.shear = {'weight': 1.0, 'reduce': 0.0, 'delta': shear_delta}
    log_quantities.append('shear_acceptance')

    # use length moves if we have a pressure, aspect moves otherwise
    if betaP != None:
        box_mc.length = {'weight': 1.0, 'delta': length_delta}
        # volume_acceptance also include the calculation of length moves
    else:
        box_mc.aspect = {'weight': 1.0, 'delta': aspect_delta}
        log_quantities.append('aspect_acceptance')

#sim.operations.updaters.append(box_mc)

box_mc_logger = hoomd.logging.Logger(categories=['scalar'])
box_mc_status = Boxmc_status(box_mc)

for i in log_quantities:
    box_mc_logger[(i)] = (box_mc_status, i, 'scalar')

box_mc_writer = hoomd.write.Table(output=open('box_mc.txt', mode='w', newline='\n'),
                               trigger=hoomd.trigger.Periodic(1000),
                               logger=box_mc_logger)
# sim.operations.writers.append(box_mc_writer)

terminate = Walltimereach(sim)
terminator = hoomd.write.CustomWriter(
    action=terminate, trigger=hoomd.trigger.Periodic(doc_gsd_frequency))
sim.operations.writers.append(terminator)

# add patchy interaction
patch_code = patch_c_code.code_patch_SQWELL.format(
    patch_locations=generate_patch_location_c_code(patch_locations),
    n_patches=len(patch_locations),
    epsilon=1 / kT,
    repulsive_epsilon=epsilon_ratio / kT,
    sigma=sigma,
    lambdasigma=lambdasigma,
    repulsive_radius=repulsive_radius,
)

patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=2, code=patch_code, param_array=[])
mc.pair_potential = patch

potential_logger = hoomd.logging.Logger(categories=['scalar'])
potential_logger.add(sim, quantities=['timestep'])
potential_logger.add(patch, quantities=['energy'])
potential_writer = hoomd.write.Table(output=open('hpmc_potential.txt', mode='w', newline='\n'),
                               trigger=hoomd.trigger.Periodic(thermo_frequency),
                               logger=potential_logger)
sim.operations.writers.append(potential_writer)

# adjust tuner
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=['a', 'd'],
    target=0.33,
    trigger=hoomd.trigger.And([
                            hoomd.trigger.Periodic(100),
                            hoomd.trigger.Before(sim.timestep + n_tune_blocks*n_tune_steps)
    ])
)

sim.operations.tuners.append(tune)

# run NVT MC
while True:
    final_timestep = compressing_timestep + doc_stop_after
    if sim.timestep >= final_timestep:
        break
    try:
        sim.run(doc_n_run_steps)
        if commute.num_ranks == 1:
            doc_timestep = sim.timestep
        commute.barrier()
        restart_writer.write(state=sim.state,
                             mode='wb',
                             filename=restart_fn,
                             log=logger)
    except Exception as exc:
        print(exc)
        restart_writer.write(state=sim.state,
                         mode='wb',
                         filename=restart_fn,
                         log=logger)
        if commute.num_ranks == 1:
            doc_timestep = sim.timestep
        commute.barrier()
        break
