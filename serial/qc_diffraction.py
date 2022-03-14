import gsd.hoomd
import freud
import matplotlib.pyplot as plt
import numpy as np
'''
trajectory = gsd.hoomd.open('trajectory.gsd')

N_frame = len(trajectory)
frame_name = np.around(np.linspace(1, N_frame-1, 10))

fig_size = 2048
dp = freud.diffraction.DiffractionPattern(grid_size=fig_size, output_size=fig_size)
plt.ioff()
fig, ax = plt.subplots(2, int(len(frame_name)/2), figsize=(10 * int(len(frame_name)/2), int(2*8)))
ax = np.ravel(ax)
for frame_idx, n in enumerate(frame_name):
    n = int(n)
    frame = trajectory[n]
    system = (freud.box.Box.from_box(frame.configuration.box), frame.particles.position)

    dp.compute(system, view_orientation=[1, 0, 0, 0])

    dp.plot(ax=ax[frame_idx])
    ax[frame_idx].set_title(f"frame {n*10000}", fontsize=25)

ax = np.reshape(ax, (2, int(len(frame_name)/2)))
plt.savefig('diffraction_pattern.jpg', dpi=500)

freud.parallel.set_num_threads(8)
'''
trajectory = gsd.hoomd.open('restart.gsd')
fig_size = 2048
dp = freud.diffraction.DiffractionPattern(grid_size=fig_size, output_size=fig_size)
frame = trajectory[-1]
system = (freud.box.Box.from_box(frame.configuration.box), frame.particles.position)

dp.compute(system, view_orientation=[1, 0, 0, 0], peak_width=1)

dp.plot()
plt.savefig('diffraction_pattern.jpg')
