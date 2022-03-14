import flow
from source import workflow, labels
from source.workflow import Project
import signac
import numpy as np

pr = signac.get_project()

N_GPU = 1
INIT_WALLTIME = 0.1
RAMP_WALLTIME = 24
SIM_WALLTIME = 48


def init_doc(job):
    job.doc.gsd_period = 10_000
    job.doc.thermo_period = 100_000
    job.doc.n_tune_steps = 2_000_000
    job.doc.n_run_blocks = 1500
    job.doc.n_run_steps = 1_000_000
    job.doc.ramp_run_steps = 10_000_000
    job.doc.scale = 0.01
    job.doc.do_tuning = True
    job.doc.do_annealing = False
    job.doc.compressed = False
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
        label='init',
        base_function=simulation.initialize_kagome_snap
    )

    # 2. Initialize simulation for compression
    sim = simulation.initialize_polygons_hpmc(
        job=job,
        sim=sim,
        gsd_period=job.doc.compress_period,
        thermo_period=job.doc.compress_period,
        t_tune_end=job.doc.n_tune_steps,
        label='init',
        kT=job.sp.kT_init,
        binary=True,
        patchy=True
    )

    # 3. Compress the system to target density
    if job.sp.initial_state == 'dilute':
        simulation.compress_run(
            job=job,
            sim=sim,
            label='init',
            t_end=int(2e6),
            run_walltime=INIT_WALLTIME * 3600,  # s
            t_block=job.doc.compress_period
        )
    else:
        job.doc['compressed'] = True
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=RAMP_WALLTIME, n_gpu=N_GPU))
@Project.pre.after(init_and_compress)
@Project.post(labels.tramp_complete)
def temp_ramp(job):
    from source import simulation
    import hoomd

    # 1. Use GPU
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)

    label = 't_ramp'
    sim = simulation.restart_sim(
        job=job,
        sim=sim,
        label='init',
    )

    # 2. Initialize simulation for compression
    sim = simulation.initialize_polygons_hpmc(
        job=job,
        sim=sim,
        gsd_period=job.doc.gsd_period,
        thermo_period=job.doc.thermo_period,
        t_tune_end=job.doc.ramp_run_steps,
        label=label,
        kT=job.sp.kT_init,
        binary=True,
        patchy=True
    )

    # 3. Compress the system to target density
    simulation.annealing_run(
        job=job,
        sim=sim,
        label=label,
        t_end=job.doc.ramp_run_steps,
        run_walltime=RAMP_WALLTIME * 3600,  # s
        temp_start=job.sp.kT_init,
        temp_end=job.sp.kT_end,
        binary=True,
        t_block=100_000
        )
    job.doc['t_ramp_done'] = True
    return

@workflow.simulation_group
@Project.operation.with_directives(
    workflow.sim_gpu_directives(walltime=SIM_WALLTIME, n_gpu=2))
@Project.pre.after(temp_ramp)
@Project.post(labels.sim_complete)
def run_simulation(job):
    from source import simulation
    import hoomd

    # 1. Use gpu
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    label = 'sampling'

    sim = simulation.restart_sim(
        job=job,
        sim=sim,
        label='t_ramp'
    )

    # 2. Initialize simualtion for self-assembly run
    sim = simulation.initialize_polygons_hpmc(
        job=job,
        sim=sim,
        gsd_period=job.doc.gsd_period,
        thermo_period=job.doc.thermo_period,
        t_tune_end=job.doc.stop_after,
        label=label,
        kT=job.sp.kT_end,
        binary=True,
        patchy=True
    )

    # 3. self-assembly run
    simulation.restartable_run(
        job=job,
        sim=sim,
        label=label,
        t_end=job.doc.stop_after,
        run_walltime=SIM_WALLTIME * 3600,  # s
        t_block=job.doc.n_run_steps
    )
    return

if __name__ == "__main__":
    Project().main()
