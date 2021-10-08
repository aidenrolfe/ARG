import sys
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import interpn
import astropy.convolution as conv
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib import pyplot as plt

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


def main():
    scale = process('input')
    process('target', scale)


def process(name, scale=None, batchsize=1000):
    images = np.load(f'{name}galaxies.npy')
    redshifts = np.load(f'{name}redshifts.npy').squeeze()
    # apply a uniform rescaling
    if scale is None:
        scale = 20000 / np.max(images)
    images *= scale
    nsplit = len(images)//batchsize
    batches = [np.array_split(x, nsplit) for x in (images, redshifts)]
    plot_images = []
    for i, (imbatch, zbatch) in enumerate(zip(*batches)):
        # batch is a "view", so this updates the contents of the images array
        imbatch[:], pltimg = observe_gals(imbatch, zbatch)
        plot_images.append(pltimg)
        print(f'Processed {name} batch {i} of {nsplit}', end='\r')
    np.save(f'{name}galaxies_obs_nonoise.npy', images)
    for i, (imbatch, zbatch) in enumerate(zip(*batches)):
        imbatch[:], pltimg = add_noise(imbatch)
        plot_images[i].update(pltimg)
        test_plot(plot_images[i], f'{name}_{i}_obs')
        print(f'Added noise to batch {i} of {nsplit}', end='\r')
    np.save(f'{name}galaxies_obs.npy', images)
    return scale


def observe_gals(images, redshifts, seeing=3.5, nominal_redshift=0.1,
                 dimming=False, plot_idx=0):
    plot_images = {"original": images[plot_idx].copy()}
    images = rebinning(images, nominal_redshift, redshifts)
    plot_images["rebinned"] = images[plot_idx]
    if dimming:
        images = dimming(images, nominal_redshift, redshifts)
        plot_images["dimming"] = images[plot_idx]   
    images = convolve_psf(images, seeing)
    plot_images["convolved"] = images[plot_idx]
    return images.astype(np.float32), plot_images


def add_noise(images, background=10, plot_idx=0):
    plot_images = {}
    images = add_shot_noise(images)
    plot_images["shot noise"] = images[plot_idx]
    images = add_background(images, background)
    plot_images["background noise"] = images[plot_idx]
    return images.astype(np.float32), plot_images


def test_plot(images, filename='redshifting.pdf', ncol=5):
    t = list(images.keys())[-1]
    c = images[t].shape[-1]
    col_idx = np.linspace(0, c-1, ncol).astype('int')
    n = len(images)
    fig, ax = plt.subplots(n, ncol, figsize=(2 * ncol, 2 * n))
    # base normalisation on final image
    img = images[t]
    vmax = img.max()
    vmin = -0.1 * vmax
    for i, t in enumerate(images):
        for j in range(ncol):
            ax[i, j].imshow(images[t][..., col_idx[j]], interpolation='nearest',
                            origin='lower', cmap='inferno',
                            vmin=vmin, vmax=vmax)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(t)
    plt.tight_layout()
    if filename:
        plt.savefig(f'{filename}.pdf')
    else:
        plt.show()
    plt.close(fig)

        
def convolve_psf(images, seeing):
    # We can apply convolution to all the images simultaneously
    # if the seeing is the same across all images
    # See tests at https://gist.github.com/19e446a494fbe5b5ad4c4384c23a55a9
    stdev = seeing / 2.354
    psf = conv.Gaussian2DKernel(stdev)
    psf = psf.array[None, ..., None]
    images = convolve(images, psf, mode='constant')
    return images


#def dimming(images, input_redshifts, output_redshifts):
#    d_i = cosmo.luminosity_distance(input_redshifts)
#    d_o = cosmo.luminosity_distance(output_redshifts)
#    dimming_factors = (d_i / d_o)**2
#    images = (images.T * dimming_factors).T
#    return images


def rebinning(images, input_redshifts, output_redshifts):
    # To rebin in the way we want (preserving the image size) required some thought
    # See tests at https://gist.github.com/4ccc4da519bba5216f58d12070865fc3
    d_i = (cosmo.luminosity_distance(input_redshifts))
    d_o = (cosmo.luminosity_distance(output_redshifts))
    scale_factors = (d_i / (1 + input_redshifts)**2) / (d_o / (1 + output_redshifts)**2)
    rebinned = zoom_contents(images, scale_factors.value, image_axes=[1, 2])
    return rebinned


def zoom_contents(image, scale, image_axes=[0, 1], method='linear', conserve_flux=True, fill_value=0):
    # Resize contents of image relative to fixed image size using interpolation
    in_coords = [np.arange(s) - (s - 1) / 2 for s in image.shape]
    out_coords = np.array(np.meshgrid(*in_coords, indexing='ij'))
    # match shape of input scale to allow broadcasting
    scale = np.atleast_1d(scale)
    scale = scale.reshape(scale.shape + (1,) * (image.ndim - scale.ndim))
    out_coords[image_axes] /= scale
    out_coords = np.transpose(out_coords)
    output = interpn(in_coords, image, out_coords,
                     method=method, bounds_error=False, fill_value=fill_value)
    output = output.T
    if conserve_flux:
        output /= scale**2
    return output
    
    
def add_shot_noise(images):    
    images = np.random.poisson(images)
    return images


def add_background(images, backgrounds):
    # Adding background noise (from numerous sources - i.e. the sky,
    # electrons in detector which are appearing randomly from thermal
    # noise). Value needs to be tested to find a suitable value to
    # determine the amount of background noise added to be realistic
    # The backgrounds can be a constant, or an array giving a different
    # value for each channel.
    noise = np.random.normal(0.0, 1.0, size=images.shape)
    images = images + backgrounds * noise
    return images


if __name__ == "__main__":
    sys.exit(main())
