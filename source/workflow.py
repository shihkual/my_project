from .labels import Project
import flow

simulation_group = Project.make_group(name='embeded_sim')

def sim_gpu_directives(walltime: float = 0.5, n_gpu: int = 1, job_workspace_parent: str = 'None'):
    hostname_pattern = flow.environment.get_environment().hostname_pattern
    mpirun_str = f'mpirun -n {n_gpu}'
    if hostname_pattern  == '.*\\.bridges2\\.psc\\.edu$':
        # Bridges 2
        directives = {
            "executable": f"module purge" +
                          f"\nmodule load openmpi/4.0.5-gcc10.2.0" +
                          f"\n{mpirun_str} singularity exec --nv /$HOME/hoomd_container/software.sif python3",
            "np": int(divmod(n_gpu, 8.001)[0] + 1),
            "ngpu": n_gpu,
            "walltime": walltime
        }

    elif hostname_pattern == '.*\\.expanse\\.sdsc\\.edu$':
        # Expanse
        if job_workspace_parent == 'None':
            raise NotImplementedError(
                    "Now Expanse requires user to specify the project directory to run in a container. Maybe fixed in the future"
                    )
        directives = {
            "executable": f"module purge" +
                          f"\nmodule load gcc slurm singularitypro" +
                          f"\n{mpirun_str} singularity exec --nv --bind {job_workspace_parent} /$HOME/hoomd_container/software_gpu.sif python3",
            "np": int(divmod(n_gpu, 4.001)[0] + 1),
            "ngpu": n_gpu,
            "walltime": walltime
        }

    elif hostname_pattern == 'gl(-login)?[0-9]+\\.arc-ts\\.umich\\.edu':

        # Greatlakes
        directives = {
            "executable": f"module purge"
                          f"\nmodule load gcc/8.2.0 openmpi/4.0.2 singularity" +
                          f"\n{mpirun_str} singularity exec --nv /$HOME/hoomd_container/software.sif python",
            "ngpu": 1,
            "walltime": walltime
        }
    else:
        raise NotImplementedError("Environment not supported")

    return directives

def sim_cpu_directives(walltime: float = 0.5, n_cpu: int = 1, job_workspace_parent: str = 'None'):
    hostname_pattern = flow.environment.get_environment().hostname_pattern
    mpirun_str = f'mpirun -n {n_cpu}'
    if hostname_pattern  == '.*\\.bridges2\\.psc\\.edu$':
        # Bridges 2
        directives = {
            "executable": f"module purge" +
                          f"\nmodule load openmpi/4.0.5-gcc10.2.0" +
                          f"\nsingularity exec /$HOME/hoomd_container/software.sif python3",
            "nranks": n_cpu,
            "walltime": walltime
        }

    elif hostname_pattern == '.*\\.expanse\\.sdsc\\.edu$':
        # Expanse
        if job_workspace_parent == 'None':
            raise NotImplementedError(
                    "Now Expanse requires user to specify the project directory to run in a container. Maybe fixed in the future"
                    )
        directives = {
            "executable": f"module purge" +
                          f"\nmodule load gcc slurm singularitypro" +
                          f"\n{mpirun_str} singularity exec --bind {job_workspace_parent} /$HOME/hoomd_container/software.sif python3",
            "nranks": n_cpu,
            "walltime": walltime
        }

    elif hostname_pattern == 'gl(-login)?[0-9]+\\.arc-ts\\.umich\\.edu':

        # Greatlakes
        directives = {
            "executable": f"module purge"
                          f"\nmodule load gcc/8.2.0 openmpi/4.0.2 singularity" +
                          f"\n singularity exec /$HOME/hoomd_container/software.sif python",
            "nranks": n_cpu,
            "walltime": walltime
        }
    else:
        raise NotImplementedError("Environment not supported")

    return directives
