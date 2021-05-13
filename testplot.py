import matplotlib.pyplot as plt
import numpy as np 


## reading galaxy images and redshifts 
gal_input = np.load('inputgalaxies.npy')
z_in = np.load('inputredshifts.npy')    
gal_target = np.load('targetgalaxies.npy')
z_out = np.load('targetredshifts.npy')
gal_input_obs = np.load('inputgalaxies_obs.npy')
gal_target_obs = np.load('targetgalaxies_obs.npy')


## plot random galaxy (input and target)
def plot_example(gal_input, gal_target, z_input, z_target, gal_input_obs, gal_target_obs):
    n, w, h, c = gal_input.shape
    fig, ax = plt.subplots(6, c, figsize=(2*c, 7))
    i = np.random.default_rng().choice(n)
    vmax = gal_input[i].max()
    vmax_obs = gal_input_obs[i].max()
    vmin_obs = -0.1 * vmax
    for j in range(c):
        ax[0][j].imshow(gal_input[i, ..., j], vmin=0, vmax=vmax)
        ax[1][j].imshow(gal_target[i, ..., j], vmin=0, vmax=vmax)
        ax[2][j].imshow(gal_input_obs[i, ..., j], vmin=vmin_obs, vmax=vmax_obs)
        ax[3][j].imshow(gal_target_obs[i, ..., j], vmin=vmin_obs, vmax=vmax_obs)
        for k in [0, 1, 2, 3]:
            ax[k, j].set_xticks([])
            ax[k, j].set_yticks([])
    ax[0][0].set_ylabel(f"input z={z_input[i]:.2f}")
    ax[1][0].set_ylabel(f"target z={z_target[i]:.2f}")
    ax[2][0].set_ylabel("input after obs.")
    ax[3][0].set_ylabel("target after obs.")
    ax_input = plt.subplot(6, 1, 5)
    ax_target = ax_input.twinx()
    ax_input.plot(np.arange(c), gal_input[i].sum(axis=(0, 1)), 'bo-', label='input')
    ax_target.plot(np.arange(c), gal_target[i].sum(axis=(0, 1)), 'ro-', label='target')
    ax_input.set_xlabel('filter number')
    ax_input.set_ylabel('input flux')
    ax_target.set_ylabel('target flux')
    ax_input.set_xlim(-0.5, c-0.5)
    ax_input = plt.subplot(6, 1, 6)
    ax_target = ax_input.twinx()
    ax_input.plot(np.arange(c), gal_input_obs[i].sum(axis=(0, 1)), 'bo-', label='input')
    ax_target.plot(np.arange(c), gal_target_obs[i].sum(axis=(0, 1)), 'ro-', label='target')
    ax_input.set_xlabel('filter number')
    ax_input.set_ylabel('input flux')
    ax_target.set_ylabel('target flux')
    ax_input.set_xlim(-0.5, c-0.5)
    plt.tight_layout()
    plt.savefig('example.pdf')

plot_example(gal_input, gal_target, z_in, z_out, gal_input_obs, gal_target_obs)
