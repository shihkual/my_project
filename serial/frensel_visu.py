import fresnel
import gsd.hoomd
import numpy as np
import signac
import rowan
import matplotlib.pyplot as plt

pr = signac.get_project()
label = 'init'
for job in pr.find_jobs():
    traj = gsd.hoomd.open(name=job.fn(f'{label}_trajectory.gsd'), mode="rb")

cpu = fresnel.Device(mode='cpu', n=1)
N_frame = -1
frame = traj[N_frame]
box = frame.configuration.box
N = frame.particles.N
orientation = frame.particles.orientation
orientation = rowan.to_axis_angle(orientation)
radian_orientation = orientation[0][:, 2] * orientation[1]
particle_types = frame.particles.typeid

colors = np.empty((N, 3))
# Color by typeid
colors[particle_types == 0] = fresnel.color.linear([.95, .2, .2]) # A type
colors[particle_types == 1] = fresnel.color.linear([0.05, .95, .95]) # B type

scene = fresnel.Scene(device=cpu)
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
#geometry.outline_width = 0.05

# use color instead of material.color
geometry.material.primitive_color_mix = 1.0
geometry.color[:] = fresnel.color.linear(colors)

# create box in fresnel
fresnel.geometry.Box(scene, box, box_radius=.07)

scene.camera = fresnel.camera.Orthographic.fit(scene)
out = fresnel.pathtrace(scene, w=1800, h=1000, light_samples=5)
fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(out[:], interpolation='lanczos')
ax.axis('off')
fig.savefig('test.jpg', dpi=300)

