import flow
from source import workflow, labels
from source.workflow import Project
import signac
import numpy as np
pr = signac.get_project()

N_GPU = 1
INIT_WALLTIME = 0.1
SIM_WALLTIME = 48

def init_doc(job):
    job.doc.gsd_period = 10_000
    job.doc.thermo_period = 100_000
    job.doc.n_tune_steps = 100_000
    job.doc.n_run_blocks = 1500
    job.doc.n_run_steps = 1_000_000
    job.doc.ramp_run_steps = 10_000
    job.doc.heated_ramp = np.linspace(0.5, 1, 6)
    job.doc.quench_ramp = [0.7, 0.38]
    job.doc.scale = 0.01
    job.doc.do_tuning = True
    job.doc.do_annealing = False
    job.doc.compressed = True
    job.doc.volume_delta = 0.001
    job.doc.shear_delta = (0.001, 0, 0)
    job.doc.aspect_delta = 0.001
    job.doc.length_delta = (0.01, 0.01, 0)
    job.doc.stop_after = 50_000_000
    job.doc.continue_running = True
    job.doc.timestep = 0
    job.doc.compress_period = 1000
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=INIT_WALLTIME, n_gpu=N_GPU))
@Project.post(labels.init_complete)
@Project.post(labels.compress_complete)
def init_and_compress(job):
    """Initialize system for simulation.
    1. Initialize system at low density.
    2. Compress system to desired start density
    """
    from source import simulation
    import hoomd

    init_doc(job)
    # 1. Use GPU
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)

    sim = simulation.restart_sim(
        job=job, 
        sim=sim,
        label='compress',
        base_function=simulation.initialize_polygons_snap
    )
    
    # 2. Initialize simulation for compression
    sim = simulation.initialize_polygons_hpmc(
        job=job, 
        sim=sim,
        gsd_period=job.doc.compress_period,
        thermo_period=job.doc.compress_period,
        label='compress',
        t_tune_end=int(job.doc.n_tune_blocks * job.doc.n_tune_steps)
    )
    
    # 3. Compress the system to target density
    simulation.compress_run(
            job=job,
            sim=sim,
            label='compress',
            t_end=int(2e6),
            run_walltime=INIT_WALLTIME * 3600,  # s
            t_block=job.doc.compress_period
    )
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=INIT_WALLTIME, n_gpu=N_GPU))
@Project.pre.after(init_and_compress)
@Project.post(labels.seeded_complete)
def seeded_system(job):
    """
    Implant a DQC seed at the center of the system
    """
    from source import simulation
    import hoomd
    import gsd.hoomd

    # 1. initailize seeded system
    previous_snap = gsd.hoomd.open(job.fn('compress_restart.gsd'))[-1]
    # store the seeded snapshot in workspace as seeded_restart.gsd
    simulation.get_seeded_snap(
            job=job, 
            previous_snap=previous_snap, 
            label='seeded'
    )
    # Use GPU
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)

    sim = simulation.restart_sim(
        job=job, 
        sim=sim,
        label='seeded',
    )

    # 2. initialize simualtion with patchy interaction
    sim = simulation.initialize_polygons_hpmc(
        job=job, 
        sim=sim,
        gsd_period=job.doc.compress_period,
        thermo_period=job.doc.compress_period,
        label='seeded',
        t_tune_end=job.doc.n_tune_blocks * job.doc.n_tune_steps,
        patchy=True
    )
    
    # 3. compress the seeded system to desried density (density may change after implant the seed)
    simulation.compress_run(
            job=job,
            sim=sim,
            label='seeded',
            t_end=int(2e6),
            run_walltime=INIT_WALLTIME * 3600,  # s
            t_block=job.doc.compress_period
    )
    job.doc['seeded'] = True
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=SIM_WALLTIME, n_gpu=1))
@Project.pre.after(init_and_compress)
@Project.post(labels.sim_complete)
def run_simulation(job):
    from source import simulation
    import hoomd
    
    label = 'seeded'
    if not job.isfile('seeded_restart.gsd'):
        label = 'compress'
    # 1. Use gpu
    init_doc(job)
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)

    sim = simulation.restart_sim(
        job=job, 
        sim=sim,
        label=label, 
    )
    
    # 2. Initialize simualtion for self-assembly run
    sim = simulation.initialize_polygons_hpmc(
        job=job, 
        sim=sim,
        gsd_period=job.doc.gsd_period,
        thermo_period=job.doc.thermo_period,
        label='sampling',
        t_tune_end=job.doc.n_tune_steps + sim.timestep,
        patchy=True
    )

    # 3. self-assembly run
    simulation.restartable_run(
        job=job,
        sim=sim,
        label='sampling',
        t_end=job.doc.stop_after,
        run_walltime=SIM_WALLTIME * 3600,  # s
        t_block=job.doc.n_run_steps
    )    
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=SIM_WALLTIME, n_gpu=1))
@Project.pre.after(init_and_compress)
@Project.post(labels.sim_complete)
def restart_simulation(job):
    from source import simulation
    import hoomd
   
    # 1. Use gpu
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    label='sampling'

    sim = simulation.restart_sim(
        job=job, 
        sim=sim,
        label='none',
        base_file=f'{label}_trajectory.gsd'
    )
    
    # 2. Initialize simualtion for self-assembly run
    sim = simulation.initialize_polygons_hpmc(
        job=job, 
        sim=sim,
        gsd_period=job.doc.gsd_period,
        thermo_period=job.doc.thermo_period,
        label=label,
        t_tune_end=job.doc.n_tune_steps + sim.timestep,
        patchy=True,
        output_mode='a'
    )

    # 3. self-assembly run
    simulation.restartable_run(
        job=job,
        sim=sim,
        label='sampling',
        t_end=job.doc.stop_after,
        run_walltime=SIM_WALLTIME * 3600,  # s
        t_block=job.doc.n_run_steps
    )    
    return

if __name__ == "__main__":
    Project().main()
