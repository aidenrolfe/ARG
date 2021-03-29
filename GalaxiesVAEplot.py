from GalaxiesVAE import *

logname = '20210328-231422'

vae.load_weights(f'weights_{logname}')

if conditional:
    reconstructions = vae.predict([gal_input_test, redshifts_test])
    z_mean, z_log_var, z = encoder.predict([gal_input_test, redshifts_test])
else:
    reconstructions = vae.predict(gal_input_test)
    z_mean, z_log_var, z = encoder.predict(gal_input_test)

for i in range(10):
    plot_example(gal_input_test, gal_target_test, reconstructions, redshifts_test,
                 filename=f'example_{i}_{logname}.pdf')

