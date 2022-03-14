import rowan
import gsd.hoomd
import numpy as np
import freud
import matplotlib.pyplot as plt
import seaborn as sns
import time

freud.parallel.set_num_threads(8)
N_polygon = 5
trajectory = gsd.hoomd.open("restart.gsd")
N_frame = len(trajectory)
cluster_trajectory_f = gsd.hoomd.open(name="cluster_polygon.gsd")

frame = trajectory[-1]
cluster_frame = cluster_trajectory_f[-1]

position = frame.particles.position
box = freud.box.Box.from_box(frame.configuration.box)

cluster_position = cluster_frame.particles.position
system_temp = freud.locality.AABBQuery(box, position)
k_atic_matrix = []
k_atic_matrix_idx = []

for (cluster_idx, pos) in enumerate(cluster_position):
    neighbor_choose_idx = list(system_temp.query(pos, {'r_min': 2, 'r_max': 3.4, 'exclude_ii': True}))  #
    neighbor_choose_idx = [b for idx, (_, b, _) in enumerate(neighbor_choose_idx)]

    if len(neighbor_choose_idx) == 12:
        k_atic_matrix_idx.append(cluster_idx)
        '''
        particle_for_katic_comp = position[neighbor_choose_idx]
        particle_for_katic_comp = np.vstack((particle_for_katic_comp, pos))

        system_compute = (box, particle_for_katic_comp)
        query_args = {'mode': 'nearest', 'num_neighbors': 12, 'exclude_ii': True}
        nl = freud.locality.AABBQuery(*system_compute).query(
        particle_for_katic_comp, query_args).toNeighborList()

        psi = freud.order.Hexatic(k=12, weighted=False)
        psi.compute(system_compute, neighbors=nl)
        order = np.absolute(psi.particle_order)
        if order[-1] > 0.5:
            k_atic_matrix.append(order[-1])
            k_atic_matrix_idx.append(cluster_idx)
        

plt.hist(k_atic_matrix, bins=50)
plt.figure(figsize=(16, 16))
plt.scatter(cluster_position[k_atic_matrix_idx, 0], cluster_position[k_atic_matrix_idx, 1])
'''

cluster_trajectory_f = gsd.hoomd.open(name="refine_large_cluster_polygon.gsd", mode='wb')
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = cluster_position[k_atic_matrix_idx].shape[0]
snapshot.particles.position = cluster_position[k_atic_matrix_idx]
snapshot.particles.typeid = [0] * cluster_position[k_atic_matrix_idx].shape[0]
snapshot.configuration.box = frame.configuration.box
snapshot.particles.types = [f"{N_polygon}_polygon"]

cluster_trajectory_f.append(snapshot)

cluster_trajectory_f.close()
