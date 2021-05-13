import matplotlib.pyplot as plt
import numpy as np 


## reading galaxy images and redshifts 
gal_input = np.load('inputgalaxies.npy')
z_in = np.load('inputredshifts.npy')    
gal_target = np.load('targetgalaxies.npy')
z_out = np.load('targetredshifts.npy')

plt.figure(figsize=(15,9))

## plot random galaxy (input and target)
def plot_example(gal_input, gal_target, z_input, z_target):
    n, w, h, c = gal_input.shape
    fig, ax = plt.subplots(3, c, figsize=(2*c, 7))
    i = np.random.default_rng().choice(n)
    vmax = gal_input[i].max()
    for j in range(c):
        ax[0][j].imshow(gal_input[i, ..., j], vmin=0, vmax=vmax, cmap='inferno')
        ax[1][j].imshow(gal_target[i, ..., j], vmin=0, vmax=vmax, cmap='inferno')
        for k in [0, 1]:
            ax[k, j].set_xticks([])
            ax[k, j].set_yticks([])
    ax[0][0].set_ylabel(f"input z={z_input[i]:.2f}")
    ax[1][0].set_ylabel(f"target z={z_target[i]:.2f}")
    ax_input = plt.subplot(3, 1, 3)
    ax_target = ax_input.twinx()
    ax_input.plot(np.arange(c), gal_input[i].sum(axis=(0, 1)), 'bo-', label='input')
    ax_target.plot(np.arange(c), gal_target[i].sum(axis=(0, 1)), 'ro-', label='target')
    ax_input.legend(loc = 'upper left')
    ax_target.legend(loc = 'upper right')
    ax_input.set_xlabel('filter number')
    ax_input.set_ylabel('input flux')
    ax_target.set_ylabel('target flux')
    ax_input.set_xlim(-0.5, c-0.5)
    plt.tight_layout()
    plt.savefig('example.pdf')

plot_example(gal_input, gal_target, z_in, z_out)