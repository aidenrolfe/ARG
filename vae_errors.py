from GalaxiesVAE import *
import matplotlib.patches as patches


logdir = f"./runs/{args.runname}" 

vae.load_weights(os.path.join(logdir, 'weights'))

if args.conditional:
    reconstructions = vae.predict([gal_input_test, redshifts_test])
    z_mean, z_log_var, z = encoder.predict([gal_input_test, redshifts_test])
else:
    reconstructions = vae.predict(gal_input_test)
    z_mean, z_log_var, z = encoder.predict(gal_input_test)

residuals = gal_target_test - reconstructions
residuals_frac = residuals / gal_target_test # works out fractional error per pixel
residuals_percent = residuals_frac*100

vmax = gal_target_test[1214].max() # select galaxy to calculate errors for

fig, ax = plt.subplots(1,3)

# create circle within which to average errors
circ0 = patches.Circle((30,30), 15,linewidth=1, edgecolor='black', 
                      facecolor=(0, 0, 0, 0)) 
circ1 = patches.Circle((30,30), 15,linewidth=1, edgecolor='black', 
                      facecolor=(0, 0, 0, 0)) 
circ2 = patches.Circle((30,30), 15,linewidth=1, edgecolor='black', 
                      facecolor=(0, 0, 0, 0)) 

im0 = ax[0].imshow(residuals[1214,:,:,4], cmap='coolwarm', 
                    origin='lower', interpolation='nearest',
                    vmin=-0.1*vmax, vmax=0.1*vmax)
ax[0].set_ylabel('residual (x10)') 
ax[0].add_patch(circ0)
im1 = ax[1].imshow(residuals_frac[1214,:,:,4], cmap='coolwarm', 
                    origin='lower', interpolation='nearest',
                    vmin=-0.1*vmax, vmax=0.1*vmax)
ax[1].set_ylabel('residual fractional error') 
ax[1].add_patch(circ1)
im2 = ax[2].imshow(residuals_percent[1214,:,:,4], cmap='coolwarm', 
                    origin='lower', interpolation='nearest',
                    vmin=-10*vmax, vmax=10*vmax)
ax[2].set_ylabel('percentage error per pixel') 
ax[2].add_patch(circ2)
fig.colorbar(im2, ax=ax.ravel().tolist(), shrink=0.6)
for ax in ax.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    
img = residuals_percent[1214,:,:,4]
# create circular mask within which to average errors
w, h = img.shape[:2]
x, y = np.meshgrid(range(w), range(h))
circ_pixels = img[(x-30)**2 + (y-30)**2 <= 15**2] # apply mask to perentage error plot
error_mean = np.mean(abs(circ_pixels))
print(error_mean)
