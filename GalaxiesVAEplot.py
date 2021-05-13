from GalaxiesVAE import *

logdir = f"./runs/{args.runname}" 

vae.load_weights(os.path.join(logdir, 'weights'))

if args.conditional:
    reconstructions = vae.predict([gal_input_test, redshifts_test])
    z_mean, z_log_var, z = encoder.predict([gal_input_test, redshifts_test])
else:
    reconstructions = vae.predict(gal_input_test)
    z_mean, z_log_var, z = encoder.predict(gal_input_test)

residuals = gal_target_test - reconstructions

for i in range(10):
    plot_example(gal_input_test, gal_target_test, reconstructions, residuals, redshifts_test,
                 filename=os.path.join(logdir, f'example_{i}.pdf'))
