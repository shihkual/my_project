import signac
import numpy as np
from source.simulation import kagome_target_density

## state points ##
n_edges = int(3)                          # n_edges
patch_offset = 1                                   # patch_offset
n_repeats = int(40)              # n_repeats
initial_state = 'non_dilute'   # initial_config
seed = 9487                            # replica
pressure = 0                      # pressure
kT_end = 0.1  # np.round(np.linspace(0.02, 0.2, 10), 2).tolist()
kT_init = 0.08
epsilon_ratio = 1 # [np.round(i, 3) for i in np.linspace(3, 5, 3)]
lambdasigma = 0.2 # [np.round(i, 3) for i in np.linspace(0.05, 0.14, 10)] # 1 generates a attractive patch size that equals to the truncated edge
repulsive_radius = 0.8
length = np.sqrt(3)
kagome_theta = [60]
patch_theta = 60 * (0.1/lambdasigma)**2
do_guest = False
guest_rescaling_factor = 1
strain = 0.01
#sp_phi = kagome_target_density(length, n_edges, kagome_theta)

def main():
    #project = signac.init_project('kagome_self_assembly')
    project = signac.get_project()
    for arg1 in kagome_theta:
        sp = {'n_edges': n_edges,
            'patch_offset': patch_offset,
            'n_repeats': n_repeats,
            'initial_state': initial_state,
            'seed': seed,
            'kT_end': kT_end,
            'kT_init': kT_init,
            'pressure': pressure,
            'epsilon_ratio': epsilon_ratio,
            'sp_phi': kagome_target_density(length, n_edges, arg1),
            'lambdasigma': lambdasigma,
            'repulsive_radius': repulsive_radius,
            'length': length,
            'kagome_theta': arg1,
            'patch_theta': patch_theta,
            'do_guest': do_guest,
            'guest_rescaling_factor': guest_rescaling_factor
            'strain': strain
        }
        job = project.open_job(sp).init()

if __name__ == '__main__':
    main()
