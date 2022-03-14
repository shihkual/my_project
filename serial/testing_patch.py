import hoomd
import numpy as np
import patch_c_code

def generate_patch_location_c_code(patch_locations):
    ret_str = ''
    for pl in patch_locations:
        ret_str += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
    return ret_str


epsilon = 1
repulsive_epsilon = 1
sigma = 1
lambdasigma = 0.4
repulsive_radius = 1
patch_angle = np.deg2rad(60)
patch_locations = [[1.0, -1.1102230246251565e-16, 0.0],
                   [-0.4999999999999997, 0.8660254037844386, 0.0],
                   [-0.5000000000000003, -0.8660254037844385, 0.0]]
kagome_angle = 120

device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=1)
sim.create_state_from_gsd(f'angle_{kagome_angle}.gsd')
mc = hoomd.hpmc.integrate.ConvexPolygon()
mc.shape['host'] = dict(vertices=np.array(patch_locations)[:, :2])
sim.operations.integrator = mc
sim.run(0)

patch_locations = np.array([[0.0, 0.0, 0.0]])
patch_code = patch_c_code.code_patch_KF_triangle.format(
    patch_locations=generate_patch_location_c_code(patch_locations),
    n_patches=len(patch_locations),
    epsilon=epsilon,
    repulsive_epsilon=repulsive_epsilon,
    sigma=sigma,
    lambdasigma=lambdasigma,
    repulsive_radius=repulsive_radius,
    patch_angle=patch_angle
)
patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=1, code=patch_code, param_array=[])
mc.pair_potential = patch

sim.run(1)
print(f'When angle between triangles is {kagome_angle}\n'
      f'potential code output {patch.energy}')
