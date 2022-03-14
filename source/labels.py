from flow import FlowProject
import gsd.hoomd
import numpy as np


class Project(FlowProject):
    pass


def get_step(job, label):
    fn = f'{label}_trajectory.gsd'
    if job.isfile(fn):
        with gsd.hoomd.open(job.fn(fn), 'rb') as traj:
            return traj[-1].configuration.step
    return -1

@Project.label
def timestep_label(job):
    if get_step(job, 'sampling') >= job.doc['stop_after']:
        return 'done running'
    else:
        ts_str = np.format_float_scientific(get_step(job, 'sampling'), precision=1,
                sign=False, exp_digits=1).replace('+', '')
        sa_str = np.format_float_scientific(job.doc['stop_after'], precision=1,
                sign=False, exp_digits=1).replace('+', '')
        return 'step: {} -> {}'.format(ts_str, sa_str)

@Project.label
def init_complete(job):
    if job.doc.get('init') is None:
        return False
    else:
        return job.doc.get('init')


@Project.label
def compress_complete(job):
    if job.doc.get('compressed') is None:
        return False
    else:
        return job.doc.get('compressed')

@Project.label
def seeded_complete(job):
    if job.doc.get('seeded') is None:
        return False
    else:
        return job.doc.get('seeded')

@Project.label
def tramp_complete(job):
    if job.doc.get('t_ramp_done') is None:
        return False
    else:
        return job.doc.get('t_ramp_done')

@Project.label
def sim_complete(job):
    if job.doc.get('sampling_ts') is None:
        return False
    else:
        return get_step(job, 'sampling') >= job.doc['stop_after']
