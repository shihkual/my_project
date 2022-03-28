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
    if get_step(job, 'sampling') >= job.doc['sampling_end']:
        return 'done running'
    else:
        ts_str = np.format_float_scientific(get_step(job, 'sampling'), precision=1,
                sign=False, exp_digits=1).replace('+', '')
        sa_str = np.format_float_scientific(job.doc['sampling_end'], precision=1,
                sign=False, exp_digits=1).replace('+', '')
        return 'step: {} -> {}'.format(ts_str, sa_str)

@Project.label
def init_complete(job, label='init'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

@Project.label
def compress_complete(job, label='compress'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

@Project.label
def seed_complete(job, label='seed'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

@Project.label
def tramp_complete(job, label='tramp'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

@Project.label
def eq_complete(job, label='eq'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

@Project.label
def sampling_complete(job, label='sampling'):
    if job.doc.get(f'{label}_done') is None:
        return False
    else:
        if job.doc[f'{label}_end'] is None:
            return job.doc[f'{label}_done']
        else:
            return get_step(job, label) >= job.doc[f'{label}_end']

