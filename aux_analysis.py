import flow
import signac
from flow import directives

import gsd.hoomd
import numpy as np
import pandas as pd
import freud
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# global constant
freud.parallel.set_num_threads(8)
LABEL = 'sampling'
TRAJECTORY_FN = f"{LABEL}_trajectory.gsd"
PLOT_BOX_QUERY = ['lx', 'ly']
START_FRAME = 0
DISPLAY_FRAMES = 200
K_ATIC = [3]

def get_frame_and_katic(job):
    trajectory = gsd.hoomd.open(job.fn(TRAJECTORY_FN))
    N_frame = len(trajectory)

    chosen_frames = np.round(np.linspace(START_FRAME, N_frame-1, DISPLAY_FRAMES)).astype(int).tolist()
    k_atic = K_ATIC
    return chosen_frames, k_atic

pr = signac.get_project()
class Project(flow.FlowProject):
    def __init__(self):
        flow.FlowProject.__init__(self)

@Project.operation
@Project.pre(lambda job: job.doc.get(f"{LABEL}_done", False))
@Project.pre.isfile(f'{LABEL}_hpmc_potential.txt')
@Project.pre.isfile(f'{LABEL}_box.txt')
# @Project.post.isfile('potential.png')
@directives(walltime=1)
def plot_potential(job):
    plt.style.use('ggplot')
    plt.ioff()
    plot_box = False
    plot_temp = False

    data = np.genfromtxt(job.fn(f'{LABEL}_hpmc_potential.txt'), dtype=float, skip_header=1)
    data = data[~np.isnan(data).any(axis=1), :]
    timesteps = np.linspace(0, (data.shape[0]-1)*job.doc.thermo_period, data.shape[0])
    df_potential = pd.DataFrame(data, columns=['timestep', 'V'])

    data = np.genfromtxt(job.fn(f'{LABEL}_box.txt'), dtype=float, skip_header=1)
    data = data[~np.isnan(data).any(axis=1), :]
    df_volume = pd.DataFrame(data[:, 0], columns=['volume'])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        timesteps,
        df_potential['V'],
        marker='o',
        linestyle='dashed',
        linewidth=3,
        color='r'
    )
    ax.set_xlabel('HPMC sweep', fontsize=25)
    plt.xticks(fontsize=20)
    ax.set_ylabel('Potential energy', fontsize=25, color='r')
    plt.yticks(fontsize=20)
    
    if plot_box:
        ax2 = ax.twinx()
        ax2.plot(
            df_potential['timestep'],
            df_volume['volume'],
            marker='o',
            linestyle='dashed',
            linewidth=3,
            color='b'
        )
        ax2.set_ylabel('Box volume', fontsize=25, color='b')   
        plt.yticks(fontsize=20)
    if plot_temp:
        import hoomd
        temperature_ramp = hoomd.variant.Ramp(
            A=job.sp.kT_init,
            B=job.sp.kT_end,
            t_start=int(job.doc.tramp_end*0.05),
            t_ramp=int(job.doc.tramp_end*0.95)
        )
        temperature = [temperature_ramp(int(i)) for i in timesteps]
        ax2 = ax.twinx()
        ax2.plot(
            timesteps,
            temperature,
            marker='o',
            linestyle='dashed',
            linewidth=3,
            color='b'
        )
        ax2.set_ylabel('Temperature', fontsize=25, color='b')   
        plt.yticks(fontsize=20)

    plt.savefig(job.fn(f"{LABEL}_potential.png"), dpi=500)
    return

@Project.operation
@Project.pre(lambda job: job.doc.get(f"{LABEL}_done", False))
@Project.pre.isfile(f'{LABEL}_hpmc_potential.txt')
@Project.pre.isfile(f'{LABEL}_box.txt')
# @Project.post.isfile('box_geometry.png')
@directives(walltime=1)
def plot_box(job):
    plt.style.use('ggplot')
    plt.ioff()

    data = np.genfromtxt(job.fn(f'{LABEL}_box.txt'), dtype=float, skip_header=1)
    data = data[~np.isnan(data).any(axis=1), :]
    timesteps = np.linspace(0, (data.shape[0]-1)*job.doc.thermo_period, data.shape[0])
    data[:, 1] = (data[:, 1] - data[0, 1]) / data[0, 1] * 100
    data[:, 2] = (data[:, 2] - data[0, 2]) / data[0, 2] * 100
    df_volume = pd.DataFrame(data, columns=['volume', 'lx', 'ly', 'lz', 'xy'])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        timesteps,
        df_volume[PLOT_BOX_QUERY[0]],
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=3,
        color='r'
    )
    ax.set_xlabel('HPMC sweep', fontsize=25)
    plt.xticks(fontsize=20)
    ax.set_ylabel('Lx strain (%)', fontsize=25, color='r')
    plt.yticks(fontsize=20)

    ax2 = ax.twinx()
    ax2.plot(
        timesteps,
        df_volume[PLOT_BOX_QUERY[1]],
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=3,
        color='b'
    )
    ax2.set_ylabel('Ly strain (%)', fontsize=25, color='b')
    plt.yticks(fontsize=20)

    if job.sp.strain > 0:
        ax.set_ylim([-0.5, job.sp.strain * 100 + 0.5])
        ax2.set_ylim([-0.5, job.sp.strain * 100 + 0.5])
    else:
        ax.set_ylim([job.sp.strain * 100 - 0.5, 0.5])
        ax2.set_ylim([job.sp.strain * 100 - 0.5, 0.5])
    plt.savefig(job.fn(f"{LABEL}_box_geometry.png"), dpi=500)
    return

@Project.operation
@Project.pre(lambda job: job.doc.get(f"{LABEL}_done", False))
@Project.pre.isfile(f'{LABEL}_hpmc_potential.txt')
@Project.pre.isfile(f'{LABEL}_box.txt')
@Project.pre.isfile(TRAJECTORY_FN)
# @Project.post.isfile('kagome_angle.png')
@directives(walltime=1)
def plot_kagome_angle(job):
    import gsd.hoomd
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    plt.ioff()

    traj = gsd.hoomd.open(job.fn(TRAJECTORY_FN))
    chosen_frame, _ = get_frame_and_katic(job)
    timesteps = np.linspace(0, (len(traj) - 1)*job.doc.gsd_period, len(traj))
    kagome_angle_mean = []
    kagome_angle_std = []

    for i in chosen_frame:
        frame = traj[i]
        mean, std = compute_kagome_angle(job, frame)
        kagome_angle_mean.append(mean)
        kagome_angle_std.append(std)

    plt.figure(figsize=(16, 8))
    plt.errorbar(
        x=timesteps[chosen_frame],
        y=kagome_angle_mean,
        yerr=kagome_angle_std,
        fmt='o--',
        elinewidth=1,
        capsize=5
    )
    plt.xticks(fontsize=20)
    plt.xlabel('HPMC sweeps', fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\theta_{Kagome}$ (degrees)', fontsize=25)
    plt.savefig(job.fn(f"{LABEL}_kagome_angle.png"), dpi=500)
    return

@Project.operation
@Project.pre(lambda job: job.doc.get(f"{LABEL}_done", False))
@Project.pre.isfile(TRAJECTORY_FN)
# @Project.post.isfile('voro.mp4')
@directives(memory='4096m')
@directives(walltime=1)
def voro_diagram(job):
    from matplotlib.colorbar import Colorbar
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.animation import FuncAnimation
    trajectory = gsd.hoomd.open(job.fn(TRAJECTORY_FN))

    def show_minkowski_structure_metrics(i):
        ax.cla()
        chosen_frame, k_atic = get_frame_and_katic(job)
        frame = trajectory[chosen_frame[i]]
        host_idx = np.where(frame.particles.typeid != 2)
        box = freud.box.Box.from_box(frame.configuration.box)
        position = frame.particles.position[host_idx]
        system = (box, position)

        voro = freud.locality.Voronoi()
        voro.compute(system)

        for k in k_atic:
            psi = freud.order.Hexatic(k=k, weighted=True)
            psi.compute(system, neighbors=voro.nlist)
            order = np.absolute(psi.particle_order)

            voro.plot(ax=ax)
            patches = ax.collections[0]
            patches.set_array(order)
            patches.set_cmap("viridis")
            patches.set_clim(0, 1)
            patches.set_alpha(0.7)
            # Remove old colorbar coloring by number of sides
            ax.figure.delaxes(ax.figure.axes[-1])
            ax_divider = make_axes_locatable(ax)
            # Add a new colorbar to the right of the main axes.
            cax = ax_divider.append_axes("right", size="7%", pad="2%")
            cbar = Colorbar(cax, patches)
            cbar.set_label(fr"$\psi'_{k}$", size=20)
            ax

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ani = FuncAnimation(
        fig,
        show_minkowski_structure_metrics,
        frames=len(get_frame_and_katic(job)[0]),
        interval=500,
        repeat=False
    )
    ani.save(job.fn(f'{LABEL}_voro.mp4'), fps=int(DISPLAY_FRAMES/10), dpi=100)

@Project.operation
@Project.pre(lambda job: job.doc.get(f"{LABEL}_done", False))
@Project.pre.isfile(TRAJECTORY_FN)
# @Project.post.isfile('mean_katic.png')
@directives(walltime=0.5)
def plot_average_katic(job):
    trajectory = gsd.hoomd.open(job.fn(TRAJECTORY_FN))
    chosen_frame, k_atic = get_frame_and_katic(job)
    averaged_katics = []

    timesteps = np.linspace(0, (len(trajectory)-1)*job.doc.gsd_period, len(trajectory))
    for i in chosen_frame:
        for k in k_atic:
            frame = trajectory[i]
            host_idx = np.where(frame.particles.typeid != 2)
            box = freud.box.Box.from_box(frame.configuration.box)
            position = frame.particles.position[host_idx]
            system = (box, position)

            voro = freud.locality.Voronoi()
            voro.compute(system)
            psi = freud.order.Hexatic(k=k, weighted=True)
            psi.compute(system, neighbors=voro.nlist)
            order = np.absolute(psi.particle_order)
            averaged_katics.append(np.mean(order))

    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    plt.plot(
        timesteps[chosen_frame],
        averaged_katics,
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=8,
    )
    plt.xlabel('HPMC sweeps', fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel(fr"$\psi'_{3}$", fontsize=25)
    plt.yticks(fontsize=20)
    plt.savefig(job.fn(f"{LABEL}_mean_katic.png"), dpi=100)

def compute_kagome_angle(job, frame):
    import freud
    freud.parallel.set_num_threads(8)
    query_dict = {'num_neighbors': 1, 'exclude_ii': True}

    host_idx = np.where(frame.particles.typeid != 2)  # ignore the guest particles
    position = frame.particles.position[host_idx]
    orientation = frame.particles.orientation[host_idx]

    vertices_local, vertices_position = _convert_to_vertices(
        position,
        orientation,
        get_patchy_polygon_config(job.sp.n_edges, job.sp.patch_offset)[1]
    )

    system_vertices = (
        freud.box.Box.from_box(frame.configuration.box),
        vertices_position
    )

    system_temp = freud.locality.AABBQuery(*system_vertices)
    nl = system_temp.query(system_vertices[1], query_dict).toNeighborList().point_indices

    kagome_angle = []
    for (vertices_idx, neighbor_idx) in enumerate(nl):
        angle = np.arccos(np.clip(np.dot(
            vertices_local[vertices_idx],
            vertices_local[neighbor_idx]),
            -1.0,
            1.0
        ))
        angle = angle * 180 / np.pi - 60
        kagome_angle.append(angle)
    kagome_angle = np.hstack(kagome_angle)
    mean = np.average(kagome_angle)
    std = np.std(kagome_angle)
    return mean, std

def _convert_to_vertices(position, orientation, vertices):
    '''
    Convert the position of polygons into the vector
    of vertices and the position of vertices
    '''
    import rowan
    vertices_position = []
    vertices_local = []
    for pos_idx in range(position.shape[0]):
        for vert in vertices:
            vertices_position.append(position[pos_idx, :] + rowan.rotate(orientation[pos_idx, :], vert))
            vertices_local.append(rowan.rotate(orientation[pos_idx, :], vert))
    vertices_position = np.array(vertices_position)
    vertices_local = np.array(vertices_local)
    return vertices_local, vertices_position

def get_patchy_polygon_config(n_edges, patch_offset):
    from scipy.spatial import ConvexHull
    n_e = n_edges
    xs = np.array([np.cos(n * 2 * np.pi / n_e) for n in range(n_e)])
    ys = np.array([np.sin(n * 2 * np.pi / n_e) for n in range(n_e)])
    zs = np.zeros_like(ys)
    vertices = np.vstack((xs, ys, zs)).T
    A_particle = ConvexHull(vertices[:, :2]).volume  # in 2D, it reduce to area
    vertices = vertices - np.mean(vertices, axis=0)
    vertex_vertex_vectors = np.roll(vertices, -1, axis=0) - vertices
    half_edge_locations = vertices + 0.5 * vertex_vertex_vectors

    f = patch_offset
    patch_locations = half_edge_locations + f * (vertices - half_edge_locations)
    return A_particle, vertices, patch_locations

if __name__ == '__main__':
    Project().main()
