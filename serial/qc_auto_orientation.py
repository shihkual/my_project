import gsd.hoomd
#import signac
import freud
import rowan
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#project = signac.get_project()

#for job in project.find_jobs({'kT': .36, 'sp_phi': .63}):
#traj_fn = job.fn('trajectory.gsd')
N_bins = 40
traj_fn = 'restart.gsd'
traj = gsd.hoomd.open(traj_fn)

N_frame = -1
frame = traj[N_frame]
box = freud.box.Box.from_box(frame.configuration.box)
system = (box, frame.particles.position)

query_args = {'mode': 'nearest', 'num_neighbors': 12, 'exclude_ii': True}
nl = freud.locality.AABBQuery(box, frame.particles.position).query(
     frame.particles.position, query_args).toNeighborList()

orders = np.zeros((frame.particles.position.shape[0], 11)).astype(complex)
for k in range(2, 13):
    psi = freud.order.Hexatic(k=k, weighted=False)
    psi.compute((box, frame.particles.position), neighbors=nl)
    orders[:, k-2] = psi.particle_order
N = frame.particles.position.shape[0]

cf = freud.density.CorrelationFunction(bins=N_bins, r_max=box.Lx/2.01)

# angle_qua = frame.particles.orientation
# angle = rowan.to_axis_angle(angle_qua)[1]
auto = []
for i in range(orders.shape[1]):
    cf.compute(
        system=system, values=orders[:, i], query_points=system[1], query_values=orders[:, i]
    )
    auto.append(cf.correlation)

#for i in range(2, 13):
#    values = np.exp(complex(0, i) * angle)
#    cf.compute(
#            system=system, values=values, query_points=system[1], query_values=values
#         )
#    auto.append(np.real(cf.correlation))
auto = np.vstack(auto)
auto = np.real(auto)
xticks_idx = np.linspace(0, N_bins-1, 5).astype(int)
xticks_label = np.round(cf.bin_centers[xticks_idx], 2)

yticks_idx = np.linspace(0, 10, 6).astype(int)
yticks_label = yticks_idx + 2
def heatmap2d(arr: np.ndarray):
    fig, ax = plt.subplots(figsize=(16, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", aspect=20, pad=0.05)

    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_label, fontsize=20)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_label, fontsize=20)
    ax.set_xlabel('r', fontsize=25)
    ax.set_ylabel('g$_{k}$(r)', fontsize=25)
    ax.set_title('Bond-orientational correlation', fontsize=30)

    im = ax.imshow(arr, cmap='bwr', vmin=-0.25, vmax=0.25)
    plt.colorbar(im, cax=cax)
    plt.show()

heatmap2d(auto)
