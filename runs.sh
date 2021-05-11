conda activate ARG
python galaxies.py
python redshifting.py

mkdir runs

python GalaxiesVAE.py standard_low-standard_high inputgalaxies targetgalaxies &> runs/standard_low-standard_high.out &
python GalaxiesVAEplot.py standard_low-standard_high inputgalaxies targetgalaxies &

python GalaxiesVAE.py standard_high-standard_low targetgalaxies inputgalaxies &> runs/standard_high-standard_low.out &
python GalaxiesVAEplot.py standard_high-standard_low targetgalaxies inputgalaxies &

python GalaxiesVAE.py noisy_low-noisy_high inputgalaxies_obs targetgalaxies_obs &> runs/noisy_low-noisy_high.out &
python GalaxiesVAEplot.py noisy_low-noisy_high inputgalaxies_obs targetgalaxies_obs

python GalaxiesVAE.py noisy_high-noisy_low targetgalaxies_obs inputgalaxies_obs &> runs/noisy_high-noisy_low.out &
python GalaxiesVAEplot.py noisy_high-noisy_low targetgalaxies_obs inputgalaxies_obs &

python GalaxiesVAE.py noisy_high-noiseless_low targetgalaxies_obs inputgalaxies_obs_nonoise &> runs/noisy_high-noiseless_low.out &
python GalaxiesVAEplot.py noisy_high-noiseless_low targetgalaxies_obs inputgalaxies_obs_nonoise &

python GalaxiesVAE.py noisy_high-standard_low targetgalaxies_obs inputgalaxies &> runs/noisy_high-standard_low.out &
python GalaxiesVAEplot.py noisy_high-standard_low targetgalaxies_obs inputgalaxies


python GalaxiesVAE.py bd-standard_low-standard_high inputbdgalaxies targetbdgalaxies &> runs/bd-standard_low-standard_high.out &
python GalaxiesVAEplot.py bd-standard_low-standard_high inputbdgalaxies targetbdgalaxies &

python GalaxiesVAE.py bd-noisy_low-noiseless_high inputbdgalaxies_obs targetbdgalaxies_obs_nonoise &> runs/bd-noisy_low-noiseless_high.out &
python GalaxiesVAEplot.py bd-noisy_low-noiseless_high inputbdgalaxies_obs targetbdgalaxies_obs_nonoise &

python GalaxiesVAE.py bd-noisy_high-noiseless_low targetbdgalaxies_obs inputbdgalaxies_obs_nonoise &> runs/bd-noisy_high-noiseless_low.out &
python GalaxiesVAEplot.py bd-noisy_high-noiseless_low targetbdgalaxies_obs inputbdgalaxies_obs_nonoise

python GalaxiesVAE.py bd-noisy_high-noiseless_noconv_low targetbdgalaxies_obs inputbdgalaxies_obs_noconv &> runs/bd-noisy_high-noiseless_noconv_low.out &
python GalaxiesVAEplot.py bd-noisy_high-noiseless_noconv_low targetbdgalaxies_obs inputbdgalaxies_obs_noconv &

python GalaxiesVAE.py bd-noisy_high-standard_low targetbdgalaxies_obs inputbdgalaxies &> runs/bd-noisy_high-standard_low.out &
python GalaxiesVAEplot.py bd-noisy_high-standard_low targetbdgalaxies_obs inputbdgalaxies &
