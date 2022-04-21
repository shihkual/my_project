import signac
from flow import FlowProject, directives, cmd
from flow import environments
import os
import numpy as np
import gsd.hoomd
import pandas as pd
import freud
import matplotlib.pyplot as plt

LABEL = 'sampling'
project = signac.get_project()
freud.parallel.set_num_threads(8)
plt.style.use('seaborn')
plt.ioff()


fig, ax = plt.subplots(figsize=(12, 12))

ax.plot(np.linspace(-5, 5), np.linspace(-5, 5), 'k-', label=r"$\nu = -1$")
ax.plot(np.linspace(-5, 5), np.linspace(-5*0.75, 5*0.75), 'k--', label=r"$\nu = -0.75$")
ax.plot(np.linspace(-5, 5), np.linspace(-5*0.5, 5*0.5), 'k:', label=r"$\nu = -0.5$")
strain_data = {}
for job in project.find_jobs():
    strain_data[job.sp.kagome_theta] = []
for job in project.find_jobs():
    select_frames = int(1e7/job.doc.thermo_period)

    data_strain = np.genfromtxt(job.fn(f'{LABEL}_box.txt'), dtype=float, skip_header=1)
    data_strain = data_strain[~np.isnan(data_strain).any(axis=1), :]
    data_strain[:, 1] = (data_strain[:, 1] - data_strain[0, 1])/data_strain[0, 1] * 100
    data_strain[:, 2] = (data_strain[:, 2] - data_strain[0, 2])/data_strain[0, 2] * 100
    try:
        strain_data[job.sp.kagome_theta].append(np.mean(data_strain[-select_frames:, 1:3], axis=0))
    except:
        pass

for a in strain_data.keys():
    strain_data[a] = np.vstack(strain_data[a])
    ax.scatter(
            strain_data[a][:, 0],
            strain_data[a][:, 1],
            label=r"$\theta_{Kagome}^{eq}$ = "+f"{a}"+r"$^{\circ}$"
    )

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.legend(fontsize=20)
ax.set_xlabel('Applied strain ($\epsilon_{x}$, %)', fontsize=25)
plt.xticks(fontsize=20)
ax.set_ylabel('Resultant strain ($\epsilon_{y}$, %)', fontsize=25)
plt.yticks(fontsize=20)
plt.savefig("poissons_ratio.png", dpi=500)

