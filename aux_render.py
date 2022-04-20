import gsd.hoomd
import numpy as np
import signac
import flow
from flow import directives
import freud
from matplotlib.ticker import ScalarFormatter
import os

freud.parallel.set_num_threads(8)
LABEL = 'sampling'
TRAJECTORY_FN = f"{LABEL}_trajectory.gsd"
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
@Project.pre.isfile(TRAJECTORY_FN)
@directives(walltime=12)
@directives(memory='4G')
def render_with_katic(job):
    import rowan
    import fresnel
    import matplotlib.pyplot as plt
    import matplotlib.colors as clrs
    from matplotlib import cm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.ioff()
    trajectory = gsd.hoomd.open(job.fn(TRAJECTORY_FN))
    chosen_frame, k_atic = get_frame_and_katic(job)

    timesteps = np.linspace(0, (len(trajectory)-1)*job.doc.thermo_period, len(trajectory))
    def plot(i):
        ax.cla()
        for k in k_atic:
            frame = trajectory[chosen_frame[i]]
            box = freud.box.Box.from_box(frame.configuration.box)
            host_idx = np.where(frame.particles.typeid != 2)
            guest_idx = np.where(frame.particles.typeid == 2)
            host_position = frame.particles.position[host_idx]
            system = (box, host_position)

            voro = freud.locality.Voronoi()
            voro.compute(system)
            psi = freud.order.Hexatic(k=k, weighted=True)
            psi.compute(system, neighbors=voro.nlist)
            order = np.absolute(psi.particle_order)

            N = frame.particles.N
            orientation = frame.particles.orientation
            orientation = rowan.to_axis_angle(orientation)
            radian_orientation = orientation[0][:, 2] * orientation[1]

            colors = np.empty((N, 3))
            # Color by typeid
            colors[host_idx] = cm.viridis(order)[:, :3]
            colors[guest_idx] = fresnel.color.linear([0.0, 0.0, 1.0])

            scene = fresnel.Scene()
            scene.background_color = fresnel.color.linear([0.75, 0.75, 0.75])
            # Spheres for every particle in the system
            geometry = fresnel.geometry.Polygon(
                scene,
                N=N,
                vertices=frame.particles.type_shapes[0]['vertices']
            )
            geometry.position[:] = frame.particles.position[:, :2]
            geometry.angle[:] = radian_orientation
            geometry.material = fresnel.material.Material(specular=0.5, roughness=0.9)
            # geometry.outline_width = 0.05

            # use color instead of material.color
            geometry.material.primitive_color_mix = 1.0
            geometry.color[:] = fresnel.color.linear(colors)

            # create box in fresnel
            fresnel.geometry.Box(scene, box, box_radius=.07)

            scene.camera = fresnel.camera.Orthographic.fit(scene)
            out = fresnel.pathtrace(scene, w=2700, h=1500, light_samples=5)
            ax.imshow(out[:], interpolation='lanczos')

            norm = clrs.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%")
            fig.colorbar(sm, cax=cax).set_label(fr"$|\psi'_{k}|$")

            ax.axis('off')

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ani = FuncAnimation(
        fig,
        plot,
        frames=len(chosen_frame),
        interval=500,
        repeat=False
    )
    ani.save('traj_movie.mp4', fps=int(DISPLAY_FRAMES / 10), dpi=300)

if __name__ == "__main__":
    Project().main()

