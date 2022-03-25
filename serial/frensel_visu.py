import fresnel
import gsd.hoomd
import numpy as np
import signac
import rowan
import matplotlib.pyplot as plt

pr = signac.get_project()

for job in pr.find_jobs():
    traj = gsd.hoomd.open(name=job.fn('trajectory.gsd'), mode="rb")

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
colors[particle_types == 0] = fresnel.color.linear([.95, 0, 0]) # A type
colors[particle_types == 1] = fresnel.color.linear([0, .95, 0]) # B type

scene = fresnel.Scene(device=cpu)
# Spheres for every particle in the system
geometry = fresnel.geometry.Polygon(
    scene,
    N=N,
    vertices=frame.particles.type_shapes[0]['vertices']
)
geometry.position[:] = frame.particles.position[:, :2]
geometry.angle[:] = radian_orientation
geometry.material = fresnel.material.Material(specualr=0.5, roughness=0.9, metal=0.2)
geometry.outline_width = 0.05

# use color instead of material.color
geometry.material.primitive_color_mix = 1.0
geometry.color[:] = fresnel.color.linear(colors)

# create box in fresnel
fresnel.geometry.Box(scene, box, box_radius=.07)

scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.rembrandt()
out = fresnel.pathtrace(scene, w=1000, h=1000, light_samples=10)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(out[:], interpolation='lanczos')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
