import freud
import rowan
import gsd.hoomd
import numpy as np

def _convert_to_vertices(position, orientation, vertices):
    import rowan
    vertices_position = []
    vertices_local = []
    for pos_idx in range(position.shape[0]):
        for vert in vertices:
            vertices_position.append(position[pos_idx, :] + rowan.rotate(orientation[pos_idx, :], vert))
            vertices_local.append(rowan.rotate(orientation[pos_idx, :], vert))
    vertices_position = np.array(vertices_position)
    vertices_local = np.array(vertices_local)
    #vertices_position = np.array(vertices_position)  # particle_idx * vertices_idx * vector value
    #vertices_position = vertices_position.reshape(int(vertices_position.shape[0] * vertices_position.shape[1]),
    #                                             vertices_position.shape[2])
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

n_edges = 3
frame = gsd.hoomd.open('restart.gsd')[-1]
freud.parallel.set_num_threads(8)
query_dict = {'num_neighbors': 1, 'exclude_ii': True}

host_idx = np.where(frame.particles.typeid != 2)  # ignore the guest particles
position = frame.particles.position[host_idx]
orientation = frame.particles.orientation[host_idx]

vertices_local, vertices_position = _convert_to_vertices(
    position,
    orientation,
    get_patchy_polygon_config(n_edges, 1)[1]
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



