import freud
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.animation import FuncAnimation
import numpy as np

voro = freud.locality.Voronoi()
trajectory = gsd.hoomd.open('trajectory.gsd')

frame = trajectory[0]

box = freud.box.Box.from_box(frame.configuration.box)
position = frame.particles.position
system = (box, position)

def get_frame_and_katic():
    frames = list(range(0, len(trajectory), 200))
    k_atic = [3]
    return frames, k_atic

def show_minkowski_structure_metrics(i):
    ax.cla()
    chosen_frame, k_atic = get_frame_and_katic()
    frame = trajectory[chosen_frame[i]]
    box = freud.box.Box.from_box(frame.configuration.box)
    position = frame.particles.position
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

def plot_average_katic(trajectory):
    chosen_frame, k_atic = get_frame_and_katic()
    averaged_katics = []
    for i in chosen_frame:
        for k in k_atic:
            frame = trajectory[i]
            box = freud.box.Box.from_box(frame.configuration.box)
            position = frame.particles.position
            system = (box, position)

            voro = freud.locality.Voronoi()
            voro.compute(system)
            psi = freud.order.Hexatic(k=k, weighted=True)
            psi.compute(system, neighbors=voro.nlist)
            order = np.absolute(psi.particle_order)
            averaged_katics.append(np.mean(order))

    plt.figure()
    plt.plot(chosen_frame, averaged_katics)
    plt.xlabel('HPMC sweeps')
    plt.ylabel(fr"$\psi'_{3}$")

psi = []
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ani = FuncAnimation(
    fig,
    show_minkowski_structure_metrics,
    frames=len(get_frame_and_katic()[0]),
    interval=500,
    repeat=False
)

plt.show()
ani.save('test.mp4', fps=2, dpi=300)


plot_average_katic(trajectory)