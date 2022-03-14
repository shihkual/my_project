import rowan
import gsd.hoomd
import numpy as np
import freud
import matplotlib.pyplot as plt
freud.parallel.set_num_threads(8)

N_polygon = 5
trajectory = gsd.hoomd.open("restart.gsd")
N_frame = len(trajectory)
cluster_trajectory_f = gsd.hoomd.open(name="cluster_polygon.gsd", mode='wb')

frame = trajectory[-1]

position = frame.particles.position
N_particles = position.shape[0]
orientation = frame.particles.orientation
box = freud.box.Box.from_box(frame.configuration.box)

n_e = 3
xs = np.array([np.cos(n * 2 * np.pi / n_e) for n in range(n_e)])
ys = np.array([np.sin(n * 2 * np.pi / n_e) for n in range(n_e)])
zs = np.zeros_like(ys)
vertices = np.vstack((xs, ys, zs)).T

vertices_position = []

for pos_idx in range(N_particles):
    one_particle = []
    for vert in vertices:
        one_particle.append(position[pos_idx, :] + rowan.rotate(orientation[pos_idx, :], vert))
    vertices_position.append(one_particle)

vertices_position = np.array(vertices_position)  # particle_idx * vertices_idx * vector value
vertices_position = vertices_position.reshape(int(vertices_position.shape[0] * vertices_position.shape[1]),
                                              vertices_position.shape[2])

system = freud.locality.LinkCell(box, vertices_position)

cl = freud.cluster.Cluster()

cl.compute(system, neighbors={"r_max": np.sqrt(3) * 0.1})

clp = freud.cluster.ClusterProperties()
clp.compute(system, cl.cluster_idx)


polygon_cluster_idx = np.where(clp.sizes == N_polygon)[0] #
polygon_clusters_position = clp.centers[clp.sizes == N_polygon]


q_cluster_idx = np.where(clp.sizes == N_polygon - 1)[0] #
q_clusters_position = clp.centers[clp.sizes == N_polygon - 1]

# polygon_cluster_to_eliminate_idx = np.floor_divide(cl.cluster_keys[polygon_cluster_idx[0]], 3) #

# particle_for_katic_comp = np.delete(position, polygon_cluster_to_eliminate_idx, axis=0)
'''
system_temp = freud.locality.AABBQuery(box, position)
neighbor_choose_idx = list(system_temp.query(polygon_clusters_position, {'r_min': 2, 'r_max': 4.5, 'exclude_ii': True})) #
neighbor_choose_idx = [j for idx, (i, j, k) in enumerate(neighbor_choose_idx)]
particle_for_katic_comp = position[neighbor_choose_idx]
particle_for_katic_comp = np.vstack((particle_for_katic_comp, polygon_clusters_position[0]))

system_compute = (box, particle_for_katic_comp)
voro = freud.locality.Voronoi()
voro.compute(system_compute)

psi = freud.order.Hexatic(k=12, weighted=True)
psi.compute(system_compute, neighbors=voro.nlist)
order = np.absolute(psi.particle_order)
'''
'''
test_position = np.vstack((polygon_clusters_position, q_clusters_position))
'''
frame.configuration.box[2] = 10
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = polygon_clusters_position.shape[0]
snapshot.particles.position = polygon_clusters_position
snapshot.particles.typeid = [0] * polygon_clusters_position.shape[0]
snapshot.configuration.box = frame.configuration.box
snapshot.particles.types = [f"{N_polygon}_polygon"]

cluster_trajectory_f.append(snapshot)

cluster_trajectory_f.close()

'''
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = test_position.shape[0]
snapshot.particles.position = test_position
snapshot.particles.typeid = [0] * polygon_clusters_position.shape[0] + [1] * q_clusters_position.shape[0]
snapshot.configuration.box = frame.configuration.box
snapshot.particles.types = [f"{N_polygon}_polygon", '4_polygon']

cluster_trajectory_f.append(snapshot)

cluster_trajectory_f.close()
'''