import signac
import gsd.hoomd
import numpy as np
import pandas as pd
from sklearn import decomposition, metrics
from sklearnex import patch_sklearn
patch_sklearn()

project = signac.get_project()
# global constant
original_trajectory = 'trajectory.gsd'
clustering_trajectory = 'cluster_polygon.gsd'
k_atic = list(range(2, 13))
N_chosen_frame = 500
start_frame = 1
particle_wise_displayed_frames = 20
cluster_displayed_frames = 20
N_polygon = 5 # number of edges of the desired polygon for the clustering analysis

def get_frame_and_katic(job):
    trajectory = gsd.hoomd.open(job.fn(original_trajectory))
    N_frame = len(trajectory)
    del trajectory

    chosen_frame = np.linspace(start_frame, int(N_frame-1), N_chosen_frame).astype(int).tolist()
    k_atic_name = k_atic
    N_k_atic = len(k_atic)
    return chosen_frame, k_atic_name, N_k_atic

for job in project.find_jobs():
    df = pd.DataFrame()
    data = []
    chosen_frame, k_atic_name, N_k_atic = get_frame_and_katic(job)
    chosen_frame = chosen_frame[0:-1:25]
    with job.data:
        for key1 in [f"{i:2.0f}_msm_abs" for i in k_atic_name]:
            temp = []
            for key2 in [f"frame_{i:5.0f}" for i in chosen_frame]:
                temp.append(job.data[f"{N_polygon}_polygon_cluster_in_origin"][key2][key1][()])
            data.append(np.hstack(temp))

    cluster_trajectory = gsd.hoomd.open(job.fn(clustering_trajectory))
    N_cluster_list = []
    for (frame_idx, frame) in enumerate(cluster_trajectory[int(i - 1)] for i in chosen_frame):
        N_cluster_list.append(frame.particles.N)
    # for creating a cumulative function of # of polygons
    N_cluster_list_cumulative = [sum(N_cluster_list[:i]) for i in range(0, int(len(N_cluster_list) + 1))]

    data = np.vstack(data)
    df = df.append(pd.DataFrame(data))
    del data, temp, N_cluster_list

    X = np.transpose(df.to_numpy()).astype('float32')
    del df
    # Values of epsilon in base 2 we want to scan.
    '''
    eps = np.power(2., np.arange(-10., 14., 1))

    # Pre-allocate array containing sum(Aij).
    Aij = np.zeros(eps.shape)

    # Loop through values of epsilon and evaluate matrix sum.
    for i in range(len(eps)):
        A = metrics.pairwise.rbf_kernel(X, gamma=1. / (2. * eps[i]))
        Aij[i] = A.sum()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))
    plt.plot(np.log(eps), np.log(Aij))
    plt.xlabel('$\log(\epsilon)$')
    plt.ylabel('$\log(\sum{A_{ij}})$')
    plt.savefig('eps.jpg', dpi=300)
    '''

    # From the plot above we see that 4 is a good choice.
    eps = np.power(2., -4)

    # Generate final matrix A, and row normalized matrix M.
    A = metrics.pairwise.rbf_kernel(X, gamma=1. / (2. * eps))
    M = A / A.sum(axis=1, keepdims=True)

    # Get the eigenvalues/vectors of M.
    # We normalize by the first eigenvector.
    W, V = np.linalg.eig(M)
    V = V / V[:, 0]

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    plt.ioff()
    '''
    plt.figure(figsize=(15, 10))
    plt.scatter(V[:, 1], V[:, 2])
    plt.xlabel('$\mathbf{\Psi}_2$')
    plt.ylabel('$\mathbf{\Psi}_3$')
    '''
    real_frame_name = np.array(chosen_frame) * job.doc.gsd_frequency
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # Choose a colormap
    colormap = cm.viridis
    normalize = mcolors.Normalize(vmin=min(real_frame_name), vmax=max(real_frame_name))

    color_map_ticks = []
    for a in range(len(chosen_frame)-1):
        color_map_ticks.append(real_frame_name[a])

    ax.set_xlabel('$\mathbf{\Psi}_2$', fontsize=15)
    ax.set_ylabel('$\mathbf{\Psi}_3$', fontsize=15)

    for a in range(len(chosen_frame)-1):
        color = colormap(normalize((real_frame_name[a])))
        ax.scatter(
            V[N_cluster_list_cumulative[a]:N_cluster_list_cumulative[a + 1], 1],
            V[N_cluster_list_cumulative[a]:N_cluster_list_cumulative[a + 1], 2],
            color=color,
            alpha=0.5)
    ax.grid()

    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(color_map_ticks)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (color_map_ticks[1] - color_map_ticks[0]) / 2.0
    boundaries = np.linspace(color_map_ticks[0] - halfdist,
                             color_map_ticks[-1] + halfdist,
                             len(color_map_ticks) + 1)

    cbar = fig.colorbar(s_map, ax=ax, spacing='proportional',
                        ticks=color_map_ticks,
                        boundaries=boundaries,
                        location='bottom',
                        shrink=0.3
                        )
    cbar.set_ticks(color_map_ticks)
    cbar.ax.set_ylabel('HPMC timestep', fontsize=15)
    fig.savefig('test.jpg', dpi=500)
