import numpy as np
import signac
import gsd.hoomd
import hoomd
import os

class Editor:
    def __init__(
            self,
            job: signac.contrib.job.Job,
            simulation: "hoomd.Simulation",
            mc: "hoomd.hpmc.integrate.HPMCIntegrator",
            label: str,
            gsd_period: int,
            thermo_period: int,
            boxmc=None,
    ):
        self.job = job
        self.sim = simulation
        self.mc = mc
        self.gsd_period = gsd_period
        self.thermo_period = thermo_period
        self.Speed_status = self.Speed_status(self.sim)
        self.Thermo_status = self.Thermo_status(self.sim)
        self.MC_status = self.MC_status(self.mc)
        if boxmc == None:
            pass
        else:
            self.boxmc = boxmc
            self.Boxmc_status = self.Boxmc_status(self.boxmc)
            self.boxmc_fn = job.fn(f'{label}_boxmc.txt')
        self.gsd_fn = job.fn(f'{label}_restart.gsd')
        self.trajectory_fn = job.fn(f'{label}_trajectory.gsd')
        self.thermo_fn = job.fn(f'{label}_box.txt')
        self.mc_fn = job.fn(f'{label}_hpmc.txt')
        self.potential_fn = job.fn(f'{label}_hpmc_potential.txt')

    class Speed_status:
        def __init__(self, sim):
            self.sim = sim
        """Provide helper properties for simulation time status."""
        @property
        def seconds_remaining(self):
            try:
                return (self.sim.final_timestep - self.sim.timestep
                       ) / self.sim.tps
            except ZeroDivisionError:
                return 0

        @property
        def etr(self):
            import datetime
            return str(datetime.timedelta(seconds=self.seconds_remaining))

    class Thermo_status():
        def __init__(self, sim):
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

    @staticmethod
    def get_shape_logger(mc: "hoomd.hpmc.integrate.HPMCIntegrator"):
        logger = hoomd.logging.Logger()
        logger.add(mc, quantities=['type_shapes'])
        return logger

    def get_tps_logger(self):
        logger = hoomd.logging.Logger(categories=["scalar", "string"])
        logger.add(self.sim, quantities=["timestep", "tps"])
        status = self.Speed_status
        logger[("Status", "etr")] = (status, "etr", "string")
        return hoomd.write.Table(
            trigger=hoomd.trigger.Periodic(self.thermo_period),
            logger=logger
        )

    def get_box_geometry_logger(self, mode='w'):
        logger = hoomd.logging.Logger(categories=['scalar'])
        status = self.Thermo_status
        logger[('volume')] = (status, 'volume', 'scalar')
        logger[('lx')] = (status, 'lx', 'scalar')
        logger[('ly')] = (status, 'ly', 'scalar')
        logger[('lz')] = (status, 'lz', 'scalar')
        logger[('xy')] = (status, 'xy', 'scalar')
        return hoomd.write.Table(
            output=open(self.thermo_fn, mode=mode, newline='\n'),
            trigger=hoomd.trigger.Periodic(self.thermo_period),
            logger=logger
        )

    def get_mc_logger(self, mode='w'):
        logger = hoomd.logging.Logger(categories=['scalar'])
        status = self.MC_status
        logger[('trans_accep')] = (status, 'trans_accep', 'scalar')
        logger[('rotat_accep')] = (status, 'rotat_accep', 'scalar')
        return hoomd.write.Table(
            output=open(self.mc_fn, mode=mode, newline='\n'),
            trigger=hoomd.trigger.Periodic(self.thermo_period),
            logger=logger
        )

    def get_boxmc_logger(
            self,
            loggable_quantities: list,
            mode='w'):
        logger = hoomd.logging.Logger(categories=['scalar'])
        status = self.Boxmc_status
        for quantities in loggable_quantities:
            logger[(quantities)] = (status, quantities, 'scalar')
        return hoomd.write.Table(
            output=open(self.boxmc_fn, mode=mode, newline='\n'),
            trigger=hoomd.trigger.Periodic(self.thermo_period),
            logger=logger
        )


    def get_potential_logger(
            self,
            potential: "hoomd.hpmc.pair.user.CPPPotential",
            mode='w'
    ):
        potential_logger = hoomd.logging.Logger(categories=['scalar'])
        potential_logger.add(self.sim, quantities=['timestep'])
        potential_logger.add(potential, quantities=['energy'])
        return hoomd.write.Table(
            output=open(self.potential_fn, mode=mode, newline='\n'),
            trigger=hoomd.trigger.Periodic(self.thermo_period),
            logger=potential_logger
        )
    
    @staticmethod
    def get_onthefly_potential_logger(
            sim: "hoomd.Simulation",
            potential: "hoomd.hpmc.pair.user.CPPPotential",
            fn: str,
            thermo_period: int,
            mode='w'
    ):
        potential_logger = hoomd.logging.Logger(categories=['scalar'])
        potential_logger.add(sim, quantities=['timestep'])
        potential_logger.add(potential, quantities=['energy'])
        return hoomd.write.Table(
            output=open(fn, mode=mode, newline='\n'),
            trigger=hoomd.trigger.Periodic(thermo_period),
            logger=potential_logger
        )

    def get_gsd_logger(
            self,
            mode='ab'
    ):
        logger = self.get_shape_logger(self.mc)
        return hoomd.write.GSD(
            filename=self.gsd_fn,
            trigger=hoomd.trigger.Periodic(self.gsd_period),
            filter=hoomd.filter.All(),
            truncate=True,
            mode=mode,
            log=logger
        )

    def get_trajectory_logger(
            self,
            mode='ab'
    ):
        logger = self.get_shape_logger(self.mc)
        return hoomd.write.GSD(
            filename=self.trajectory_fn,
            trigger=hoomd.trigger.Periodic(self.gsd_period),
            filter=hoomd.filter.All(),
            mode=mode,
            log=logger
        )

class Seed:
    def __init__(self, seed_file, job, origin=np.array([0, 0, 0])):
        pr = signac.get_project()
        self.seed_fn = pr.fn(seed_file)
        self.origin = origin
        self.vertices = np.array(job.doc.vertices)
        self.seed_traj = self._get_traj()
        del pr

    def _get_traj(self):
        seed_trajectory = gsd.hoomd.open(self.seed_fn)[0]
        return seed_trajectory

    @staticmethod
    def find_overlap_idx(target_position, convexhull, tolerance=1e-9):
        in_idx = np.arange(target_position.shape[0])
        for plane in convexhull.equations:
            truth = (np.dot(
                target_position[in_idx, 0:2],
                np.reshape(plane[0:2], (2, 1))) + plane[2] <= tolerance).flatten()
            in_idx = in_idx[truth]
        return in_idx

    def _convert_to_vertices(self, position, orientation, vertices):
        import rowan
        vertices_position = []
        for pos_idx in range(position.shape[0]):
            for vert in vertices:
                vertices_position.append(position[pos_idx, :] + rowan.rotate(orientation[pos_idx, :], vert))
        vertices_position = np.array(vertices_position)  # (particle_idx * vertices_idx) * vector components
        return vertices_position

    def implant_seed(self, snap):
        from scipy.spatial import ConvexHull
        seed_position = self.seed_traj.particles.position
        seed_position = seed_position - np.mean(seed_position, axis=0)
        seed_position = seed_position - self.origin
        seed_orientation = self.seed_traj.particles.orientation
        vertices = self.vertices
        vertices_per_particles = self._convert_to_vertices(seed_position, seed_orientation, vertices)

        target_position = snap.particles.position
        target_orientation = snap.particles.orientation

        overlap_with_seed_idx = self.find_overlap_idx(
            target_position,
            ConvexHull(1.02 * vertices_per_particles[:, :2])
        )

        overlap_with_seed_filter = np.ones(target_position.shape[0]).astype(bool)
        overlap_with_seed_filter[overlap_with_seed_idx] = False

        carved_target_position = target_position[overlap_with_seed_filter]
        carved_target_orientation = target_orientation[overlap_with_seed_filter]
        carved_target_typeid = snap.particles.typeid[overlap_with_seed_filter]
        N_particles = len(carved_target_position) + len(seed_position)
        typeid_list = np.hstack((carved_target_typeid, self.seed_traj.particles.typeid))

        snapshot = gsd.hoomd.Snapshot()
        snapshot.particles.N = N_particles
        snapshot.particles.position = np.vstack((carved_target_position, seed_position))
        snapshot.particles.orientation = np.vstack((carved_target_orientation, seed_orientation))
        snapshot.particles.typeid = typeid_list
        snapshot.particles.types = snap.particles.types
        snapshot.particles.type_shapes = snap.particles.type_shapes
        snapshot.configuration.box = snap.configuration.box
        snapshot.configuration.step = snap.configuration.step
        return snapshot

def initialize_polygons_snap(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
):
    import math

    device = sim.device

    A_particle, vertices, patch_locations = get_patchy_polygons_configuration(job)
    ## build simualtion cell ##
    n_repeats = job.sp.n_repeats
    spacing = 2
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
    snapshot.particles.type_shapes = [{
        'type': 'Polygon',
        'rounding_radius': 0,
        'vertices': vertices[:, :2].tolist()
    }]
    sim.create_state_from_snapshot(snapshot)

    if device.communicator.rank == 0:
        job.doc.type_list = ['host']
        job.doc.A_particle = A_particle
        job.doc.vertices = vertices
        job.doc.patch_locations = patch_locations
        job.doc['init'] = True
    device.communicator.barrier()
    return sim

def initialize_kagome_snap(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
):
    import math
    import rowan
    from numpy import cos, sin, deg2rad

    device = sim.device
    guest = job.sp.do_guest

    ## build simualtion cell ##
    if job.sp.initial_state == 'non_dilute':  # only build unit cell for KL
        N = 2
        spacing = 1.001
        length = job.sp.length * spacing
        theta = job.sp.kagome_theta
        # side length of rhombus unit cell
        L = length * np.sqrt(2 - 2 * cos(np.pi / 3 + deg2rad(theta)))
        # unit cell vectors
        a1 = L * np.array([1, 0, 0])
        a2 = L * np.array([cos(deg2rad(60)), sin(deg2rad(60)), 0])
        a3 = np.array([0, 0, 1])

        s = length * np.sqrt(3) / 3
        # Derivation:
        # Known that for an equilateral triangle with side length l, the height h = l*sqrt(3)/2
        # Known that the distance from the side to the center along the radius of the inscribed circle
        # is 1/3 of the height, so the remainder from the center to corner is 2/3 of the height.

        ################################################################################
        # Construct with y axis aligned with a2, then rotate all coordinates.
        # In a square box, I confirmed that these work.
        angle_y_center = deg2rad(theta / 2 + 30)
        original_pos_1 = s * np.array([sin(angle_y_center), cos(angle_y_center), 0])
        original_pos_2 = s * np.array([-sin(angle_y_center), cos(angle_y_center), 0])

        # I had the neutral orientation as resting on the x axis (vertex at (0,1))
        # converting to format where one vertex is at (1,0). Otherwise +/- 30
        original_orientation_1 = deg2rad(-theta / 2)  # other convention: +30 - theta/2
        original_orientation_2 = deg2rad(60 + theta / 2)  # other convention, -30 + theta/2
        # The above are non-final coordinates and orientations
        ################################################################################

        # Now rotate coordinates:
        # x' and y' are x and y rotated by 30 degress anticlockwise, so
        # rotate points opposite direction, hence -30
        rot = rowan.from_axis_angle(a3, deg2rad(-30))
        pos_1 = rowan.rotate(rot, original_pos_1)
        pos_2 = rowan.rotate(rot, original_pos_2) + a1
        prime_positions = np.array([pos_1, pos_2])

        # convert so that the neutral orientation is resting on the x' axis
        prime_quats = [rowan.from_axis_angle(a3, t)
                       for t in [original_orientation_1 - deg2rad(30),
                                 original_orientation_2 - deg2rad(30)]]

        Lx = a1[0]
        Ly = a2[1]
        xy = a2[0] / a2[1]
        position = prime_positions - np.mean(prime_positions, axis=0)
        orientation = [tuple(list(element)) for element in prime_quats]
        typeid_list = [0, 1]
        type_list = ['host_1', 'host_2']

        if guest == True:
            N += 1
            typeid_list.append(2)
            type_list.append('guest')
            position.append((Lx / 2 - 0.0001, 0, 0))
            orientation.append((0, 0, 0, 1))

    elif job.sp.initial_state == 'dilute':  # build the whole supercell
        n_repeats = job.sp.n_repeats
        spacing = 2
        N = int(n_repeats ** 2)
        K = math.ceil(N ** (1 / 2))
        L = K * spacing
        position = []
        for i in np.linspace(-L / 2, L / 2, K, endpoint=False):
            for j in np.linspace(-L / 2, L / 2, K, endpoint=False):
                position.append((i, j, 0))
        Lx = L
        Ly = L
        xy = 0
        orientation = [(1, 0, 0, 0)] * N
        typeid_list = np.array([0] * N)
        type_list = ['host_1', 'host_2']

        if not guest:
            swap_idx = np.arange(N)
            np.random.shuffle(swap_idx)
            typeid_list[swap_idx[-int(N/2):]] = 1
        else:
            swap_idx = np.arange(N)
            np.random.shuffle(swap_idx)
            typeid_list[swap_idx[-int(N / 3):]] = 2
            typeid_list[swap_idx[-int(2 * N / 3): -int(N / 3)]] = 1
            type_list.append('guest')
    else:
        raise NotImplementedError('Unknown initial state for kagome lattice')

    A_particle, vertices, patch_locations = get_patchy_polygons_configuration(job)
    patch_direction_rotate_angle = deg2rad((120 - job.sp.kagome_theta) / 2)
    rotation_1 = rowan.from_axis_angle(np.array([0, 0, 1]), -patch_direction_rotate_angle)
    rotation_2 = rowan.from_axis_angle(np.array([0, 0, 1]), patch_direction_rotate_angle)
    patch_directions_1 = np.zeros((3, 3))
    patch_directions_2 = np.zeros((3, 3))
    for i in range(3):
        dir_norm = np.linalg.norm(vertices[i, :], 2)
        ptch_dir = vertices[i, :] / dir_norm
        patch_directions_1[i, :] = rowan.rotate(rotation_1, ptch_dir)
        patch_directions_2[i, :] = rowan.rotate(rotation_2, ptch_dir)

    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = N
    snapshot.particles.position = position
    snapshot.particles.orientation = orientation
    snapshot.particles.typeid = typeid_list
    snapshot.configuration.box = [Lx, Ly, 0, xy, 0, 0]
    snapshot.particles.types = type_list
    if not guest:
        snapshot.particles.type_shapes = [{
                'type': 'Polygon',
                'rounding_radius': 0,
                'vertices': vertices[:, :2].tolist()
        }] * 2
    else:
        snapshot.particles.type_shapes = [{
            'type': 'Polygon',
            'rounding_radius': 0,
            'vertices': vertices[:, :2].tolist()
        }] * 3

    sim.create_state_from_snapshot(snapshot)
    if job.sp.initial_state == 'non_dilute':
        sim.state.replicate(job.sp.n_repeats, job.sp.n_repeats, 1)

    if device.communicator.rank == 0:
        job.doc.type_list = type_list
        job.doc.vertices = vertices
        job.doc.patch_locations = patch_locations
        job.doc.patch_directions_1 = patch_directions_1
        job.doc.patch_directions_2 = patch_directions_2
        job.doc.A_particle = A_particle
        job.doc['init'] = True
    device.communicator.barrier()
    return sim

def initialize_polygons_hpmc(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        gsd_period: int,
        thermo_period: int,
        t_tune_end: list,
        label: str,
        output_mode: str = 'w',
        kT: float = None,
        binary: bool = False,
        patchy: bool = False,
        do_boxmc: bool = False,
        boxmc_isotropic: bool = True,
        floppy_box_move: bool = True
) -> 'hoomd.Simulation':
    vertices = np.array(job.doc.vertices)[:, :2]
    mc = hoomd.hpmc.integrate.ConvexPolygon()
    
    for type in job.doc.type_list:
        mc.shape[type] = dict(vertices=vertices)
        if type == 'guest':
            mc.shape[type] = dict(vertices=vertices * job.sp.guest_rescaling_factor)
    sim.operations.integrator = mc

    if do_boxmc:
        volume_delta = job.doc.volume_delta
        length_delta = job.doc.length_delta
        shear_delta = job.doc.shear_delta
        aspect_delta = job.doc.aspect_delta

        if job.sp.pressure != None:
            betaP = job.sp.pressure / kT
        else:
            raise NotImplementedError(
                "Intend to do the box MC simulation without providing pressure"
            )
        boxmc = hoomd.hpmc.update.BoxMC(betaP=betaP, trigger=hoomd.trigger.Periodic(10))
        boxmc_quantities = []#'betaP']
        boxmc_tune_quantities = []

        if boxmc_isotropic:
            boxmc.volume = {'mode': 'ln', 'weight': 1.0, 'delta': volume_delta}
            if np.abs(volume_delta - 0.0) > 1e-9:
                boxmc_tune_quantities.append('volume')
                boxmc_quantities.append('volume_acceptance')
        else:
            boxmc.length = {'weight': 1.0, 'delta': length_delta}
            counting = 0
            if np.abs(length_delta[0] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('length_x')
                counting += 1
            if np.abs(length_delta[1] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('length_y')
                counting += 1
            if np.abs(length_delta[2] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('length_z')
                counting += 1
            if counting > 0:
                boxmc_quantities.append('volume_acceptance')

        if floppy_box_move:
            boxmc.shear = {'weight': 1.0, 'reduce': 0.0, 'delta': shear_delta}
            counting = 0
            if np.abs(shear_delta[0] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('shear_x')
                counting += 1
            if np.abs(shear_delta[1] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('shear_y')
                counting += 1
            if np.abs(shear_delta[2] - 0.0) > 1e-9:
                boxmc_tune_quantities.append('shear_z')
                counting += 1
            if counting > 0:
                boxmc_quantities.append('shear_acceptance')
            boxmc.aspect = {'weight': 1.0, 'delta': aspect_delta}
            if np.abs(aspect_delta - 0.0) > 1e-9:
                boxmc_tune_quantities.append('aspect')
                boxmc_quantities.append('aspect_acceptance')
        
        if sim.device.communicator.rank == 0:
            print("Tune the BoxMC trial move size, including", boxmc_tune_quantities)
        sim.operations.updaters.append(boxmc)

    if patchy:
        patchy_interaction = get_patchy_interaction(job, binary=binary, kT=kT)
        mc.pair_potential = patchy_interaction

    sim.run(0)
    if do_boxmc:
        editor = Editor(job, sim, mc, label, gsd_period, thermo_period, boxmc=boxmc)
        boxmc_writer = editor.get_boxmc_logger(loggable_quantities=boxmc_quantities, mode=output_mode)
        sim.operations.writers.append(boxmc_writer)
    else:
        editor = Editor(job, sim, mc, label, gsd_period, thermo_period)
    gsd_writer = editor.get_gsd_logger()
    trajectory_writer = editor.get_trajectory_logger()
    mc_writer = editor.get_mc_logger(mode=output_mode)
    thermo_writer = editor.get_box_geometry_logger(mode=output_mode)
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(trajectory_writer)
    sim.operations.writers.append(mc_writer)
    sim.operations.writers.append(thermo_writer)
    if patchy:
        potential_writer = editor.get_potential_logger(mc.pair_potential, mode=output_mode)
        sim.operations.writers.append(potential_writer)

    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=['a', 'd'],
        target=0.33,
        max_rotation_move=np.pi/3,
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(t_tune_end[0])
        ])
    )
    sim.operations.tuners.append(tune)
    if do_boxmc:
        box_tune = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
            boxmc=boxmc,
            moves=boxmc_tune_quantities,
            target=0.33,
            trigger=hoomd.trigger.And([
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(t_tune_end[1])
            ])
        )
        sim.operations.tuners.append(box_tune)

    if output_mode == 'w':
        shape_logger = editor.get_shape_logger(mc)
        hoomd.write.GSD.write(
            state=sim.state,
            mode='wb',
            filename=job.fn(f"{label}_restart.gsd"),
            log=shape_logger
        )
        hoomd.write.GSD.write(
            state=sim.state,
            mode='wb',
            filename=job.fn(f"{label}_trajectory.gsd"),
            log=shape_logger
        )
    mc_writer.write()
    thermo_writer.write()
    if patchy:
        potential_writer.write()
    if do_boxmc:
        boxmc_writer.write()
    return sim

def compress_run(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        label: str,
        t_end: int,
        run_walltime: float,
        t_block: int = None,
) -> None:
    if t_block is None:
        t_block = t_end
    shape_logger = Editor.get_shape_logger(sim.operations.integrator)
    device = sim.device
    compressor = get_compressor(sim, job, t_block)
    sim.operations.updaters.append(compressor)
    current_walltime = 0
    try:
        while not compressor.complete and sim.timestep < t_end:
            sim.run(min(t_block, t_end - sim.timestep))
            current_walltime += sim.walltime
            if (current_walltime >= run_walltime):
                break
        if not compressor.complete:
            raise RuntimeError("Compression failed to complete")
        else:
            if device.communicator.rank == 0:
                job.doc[f'{label}_ts'] = sim.timestep
                job.doc['compressed'] = True
            hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )
            device.communicator.barrier()
    finally:
        #  Write the state of the system to a restart file
        hoomd.write.GSD.write(
            state=sim.state,
            mode='wb',
            filename=job.fn(f"{label}_restart.gsd"),
            log=shape_logger
        )

        if device.communicator.rank == 0:
            print(
                'box compression job for'
                f'{job.id} ended on steps {sim.timestep} '
                f'after {current_walltime} seconds'
            )
    del compressor    
    return sim

def deformation_run(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        label: str,
        t_end: int,
        run_walltime: float,
        t_block: int = None,
        axis: int = 0,
        buffer_time: int = 100_000
) -> 'hoomd.Simulation':
    t_actual_end = t_end + buffer_time
    if t_block is None:
        t_block = t_actual_end
    shape_logger = Editor.get_shape_logger(sim.operations.integrator)
    device = sim.device
    deformator = get_box_deformator(sim, job, t_end, axis=axis, buffer_time=buffer_time)
    sim.operations.updaters.append(deformator)
    current_walltime = 0
    try:
        # Loop until the simulation reaches the target hpmc sweeps.
        while sim.timestep < t_actual_end:
            # Run the simulation with t_block sweeps
            sim.run(min(t_block, t_actual_end - sim.timestep))
            current_walltime += sim.walltime
            #  Write the state of the system to a restart file
            if device.communicator.rank == 0:
                job.doc[f'{label}_ts'] = sim.timestep
            hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )
            device.communicator.barrier()
            # End the workflow step early if the next run would exceed the
            # alotted walltime. Use the walltime of the current run as
            # an estimate for the next.
            if (current_walltime >= run_walltime):
                break
    finally:
        #  Write the state of the system to a restart file
        hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )

        if device.communicator.rank == 0:
            print(
                f'{job.id} finish deformation on steps {sim.timestep} '
                f'after {current_walltime} seconds'
            )
    del deformator
    sim.operations.updaters.pop(-1)
    return sim

def annealing_run(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        label: str,
        t_end: int,
        run_walltime: int,
        temp_start: float,
        temp_end: float,
        binary: bool = False,
        t_block: int = None
) -> None:
    shape_logger = Editor.get_shape_logger(sim.operations.integrator)
    device = sim.device
    temperature_ramp = hoomd.variant.Ramp(
        A=temp_start,
        B=temp_end,
        t_start=int(t_end*0.05),
        t_ramp=int(t_end*0.95)
    )
    current_walltime = 0
    if t_block is None:
        t_block = t_end
    try:
        # Loop until the simulation reaches the target hpmc sweeps.
        while sim.timestep < t_end:
            # adjust the temperature
            patchy_interaction = get_patchy_interaction(
                job,
                binary=binary,
                kT=temperature_ramp(sim.timestep)
            )
            sim.operations.integrator.pair_potential = patchy_interaction

            potential_writer = Editor.get_onthefly_potential_logger(
                    sim=sim,
                    potential=sim.operations.integrator.pair_potential,
                    fn=job.fn(f'{label}_hpmc_potential.txt'),
                    thermo_period=job.doc.thermo_period,
                    mode='a'
            )
            sim.operations.writers.pop(-1)  # Always append the potential writer at the end
            sim.operations.writers.append(potential_writer)

            # Run the simulation with t_block sweeps
            sim.run(min(t_block, t_end - sim.timestep))
            current_walltime += sim.walltime
            #  Write the state of the system to a restart file
            if device.communicator.rank == 0:
                job.doc[f'{label}_ts'] = sim.timestep
            hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )
            device.communicator.barrier()
            # End the workflow step early if the next run would exceed the
            # alotted walltime. Use the walltime of the current run as
            # an estimate for the next.
            if (current_walltime >= run_walltime):
                break
    finally:
        #  Write the state of the system to a restart file
        hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )

        if device.communicator.rank == 0:
            print(
                f'{job.id} ended on steps {sim.timestep} '
                f'after {current_walltime} seconds'
            )

def restartable_run(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        label: str,
        t_end: int,
        run_walltime: int,
        t_block: int = None
) -> None:
    shape_logger = Editor.get_shape_logger(sim.operations.integrator)
    device = sim.device
    current_walltime = 0
    if t_block is None:
        t_block = t_end
    try:
        # Loop until the simulation reaches the target hpmc sweeps.
        while sim.timestep < t_end:
            # Run the simulation with t_block sweeps
            sim.run(min(t_block, t_end - sim.timestep))
            current_walltime += sim.walltime
            #  Write the state of the system to a restart file
            if device.communicator.rank == 0:
                job.doc[f'{label}_ts'] = sim.timestep
            hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )
            device.communicator.barrier()
            # End the workflow step early if the next run would exceed the
            # alotted walltime. Use the walltime of the current run as
            # an estimate for the next.
            if (current_walltime >= run_walltime):
                break
    finally:
        #  Write the state of the system to a restart file
        hoomd.write.GSD.write(
                state=sim.state,
                mode='wb',
                filename=job.fn(f"{label}_restart.gsd"),
                log=shape_logger
            )

        if device.communicator.rank == 0:
            print(
                f'{job.id} ended on steps {sim.timestep} '
                f'after {current_walltime} seconds'
            )

def restart_sim(
        job: 'signac.contrib.job.Job',
        sim: 'hoomd.Simulation',
        label: str,
        base_file: str = None,
        base_function: 'callable' = None
) -> 'hoomd.Simulation':
    '''
    This function will import a trajectory into the "hoomd.Simualtion" object from a gsd file
    and return "hoomd.Simualtion" object
    '''
    if job.isfile(f"{label}_restart.gsd"):
        f_restart = job.fn(f"{label}_restart.gsd")
        f_traj = job.fn(f"{label}_trajectory.gsd")
        restart_ts = gsd.hoomd.open(f_restart)[-1].configuration.step
        traj_ts = gsd.hoomd.open(f_traj)[-1].configuration.step
        if restart_ts < traj_ts:
            print(f'Start from the last frame in {label}_trajectory file')
            f_restart = f_traj

        sim.timestep = 0
        sim.create_state_from_gsd(f_restart, frame=-1)
        assert sim.timestep == 0
    else:
        if base_file is not None:
            sim.create_state_from_gsd(job.fn(base_file), frame=-1)
        elif base_function is not None:
            base_function(job, sim)
        else:
            raise FileNotFoundError(
                    f"The restartable file {label}_restart.gsd could not be found for {job.id}."
                )
    return sim

def get_compressor(simulation: "hoomd.Simulation", job: signac.contrib.job.Job, compress_period):
    # determine the target box volume
    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        types = 0
        N_types = {ptype: 0 for ptype in snapshot.particles.types}
        for p in snapshot.particles.types:
            n_temp = sum(snapshot.particles.typeid == types)
            N_types[p] = n_temp
        job.doc.N_types = N_types
    A_particles = 0
    for ptype, count in N_types.items():
        if ptype != 'guest':
            A_particles += job.doc.A_particle * count
    A_target = A_particles / job.sp.sp_phi
    initial_box = simulation.state.box
    final_box = hoomd.Box.from_box(initial_box)
    final_box.volume = A_target
    snapshot.communicator.barrier()

    # add compressor
    divide_compress_freq = int(compress_period/10)
    scale = job.doc.scale
    return hoomd.hpmc.update.QuickCompress(
        trigger=hoomd.trigger.Periodic(divide_compress_freq),
        target_box=final_box,
        min_scale=scale
    )

def get_box_deformator(
        simulation: "hoomd.Simulation",
        job: signac.contrib.job.Job,
        t_end: int,
        axis: int,
        buffer_time: int=100_000
):
    initial_box = simulation.state.box
    final_box = hoomd.Box.from_box(initial_box)
    scaling_factor = np.ones(3)
    scaling_factor[axis] = scaling_factor[axis] + job.sp.strain
    final_box.L = np.array([initial_box.Lx, initial_box.Ly, initial_box.Lz]) * scaling_factor
    return hoomd.update.BoxResize(
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(simulation.timestep + buffer_time + t_end)
        ]),
        box1=initial_box,
        box2=final_box,
        variant=hoomd.variant.Ramp(
                A=0,
                B=1,
                t_start=buffer_time, 
                t_ramp=t_end
        ),
        filter=hoomd.filter.All()
    )

def get_patchy_polygons_configuration(
        job: signac.contrib.job.Job
):
    '''
    Figure out the configuration of patchy polygons
    The function will return a tuple of (Area of polygons, vectors
    of vertices, vectors of patch locations)
    '''
    from scipy.spatial import ConvexHull
    n_e = job.sp.n_edges
    xs = np.array([np.cos(n * 2 * np.pi / n_e) for n in range(n_e)])
    ys = np.array([np.sin(n * 2 * np.pi / n_e) for n in range(n_e)])
    zs = np.zeros_like(ys)
    vertices = np.vstack((xs, ys, zs)).T
    A_particle = ConvexHull(vertices[:, :2]).volume  # in 2D, it reduce to area
    vertices = vertices - np.mean(vertices, axis=0)
    vertex_vertex_vectors = np.roll(vertices, -1, axis=0) - vertices
    half_edge_locations = vertices + 0.5 * vertex_vertex_vectors

    f = job.sp.patch_offset
    patch_locations = half_edge_locations + f * (vertices - half_edge_locations)
    return A_particle, vertices, patch_locations

def get_seeded_snap(
        job: 'signac.contrib.job.Job',
        previous_snap: 'hoomd.Simulation',
        label: str
) -> 'hoomd.Simulation':
    seeded_snap = Seed('source/seed.gsd', job).implant_seed(previous_snap)
    with gsd.hoomd.open(name=job.fn(f'{label}_restart.gsd'), mode='wb') as f:
        f.append(seeded_snap)
    with gsd.hoomd.open(name=job.fn(f'{label}_trajectory.gsd'), mode='wb') as f:
        f.append(seeded_snap)
    return

def kagome_target_density(edge_length, n_deges, kagome_angle):
    from numpy import cos, sin, deg2rad
    from scipy.spatial import ConvexHull

    ## calculate box area ##
    N = 2
    spacing = 1.001
    length = edge_length * spacing
    theta = kagome_angle
    # side length of rhombus unit cell
    L = length * np.sqrt(2 - 2 * cos(np.pi / 3 + deg2rad(theta)))
    A_box = L ** 2 * sin(deg2rad(60))

    ## calculate particles area ##
    n_e = n_deges
    xs = np.array([np.cos(n * 2 * np.pi / n_e) for n in range(n_e)])
    ys = np.array([np.sin(n * 2 * np.pi / n_e) for n in range(n_e)])
    zs = np.zeros_like(ys)
    vertices = np.vstack((xs, ys, zs)).T
    A_particle = ConvexHull(vertices[:, :2]).volume  # in 2D, it reduce to area

    density = N * A_particle / A_box
    return density

def get_patchy_interaction(
        job: signac.contrib.job.Job,
        kT: float = None,
        binary: bool = False
):
    if kT == None:
        kT = job.sp.kT

    patch_information = generate_patch_location_c_code(job, binary)
    if not binary:
        from .patch_c_code import code_patch_SQWELL
        patch_code = code_patch_SQWELL.format(
            patch_locations=patch_information,
            n_patches=len(job.doc.patch_locations),
            epsilon=1 / kT,
            repulsive_epsilon=job.sp.epsilon_ratio / kT,
            sigma=job.sp.sigma,
            lambdasigma=job.sp.lambdasigma,
            repulsive_radius=job.sp.repulsive_radius,
        )
        return hoomd.hpmc.pair.user.CPPPotential(
            r_cut=2.0 + 2 * job.sp.lambdasigma,
            code=patch_code,
            param_array=[]
        )
    else:
        from .patch_c_code import code_patch_KF_triangle
        patch_code = code_patch_KF_triangle.format(
            patch_locations=patch_information[0],
            patch_directions_1=patch_information[1],
            patch_directions_2=patch_information[2],
            n_patches=len(job.doc.patch_locations),
            epsilon=1 / kT,
            repulsive_epsilon=job.sp.epsilon_ratio / kT,
            lambdasigma=job.sp.lambdasigma,
            repulsive_radius=job.sp.repulsive_radius,
            patch_angle=np.cos(np.deg2rad(job.sp.patch_theta/2))
        )
        return hoomd.hpmc.pair.user.CPPPotential(
            r_cut=2.0 + 2 * job.sp.lambdasigma,
            code=patch_code,
            param_array=[]
            )

def generate_patch_location_c_code(
        job: signac.contrib.job.Job,
        binary: bool = False
):
    if not binary:
        ret_str = ''
        for pl in job.doc.patch_locations:
            ret_str += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
        return ret_str
    else:
        ptch_loc = ''
        ptch_dir1 = ''
        ptch_dir2 = ''
        for pl in job.doc.patch_locations:
            ptch_loc += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
        for pl in job.doc.patch_directions_1:
            ptch_dir1 += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
        for pl in job.doc.patch_directions_2:
            ptch_dir2 += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
        return ptch_loc, ptch_dir1, ptch_dir2
