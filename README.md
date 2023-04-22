 
[<img src="https://img.shields.io/badge/astro--ph.CO-%20%09arXiv%3A2104.00595-red.svg">](https://arxiv.org/abs/2104.00595)
      
# Cosmological Reconstructions with Artificial Neural Networks (CRANN)

*This repository is under construction.*.

In arxiv_notebooks folder, there are the notebooks that generates the figures of the paper.

If you use this code please cite our paper:
	
	Gómez-Vargas, I., Medel-Esquivel, R., García-Salcedo, R. et al. Neural network reconstructions for the Hubble parameter, growth rate and distance modulus. Eur. Phys. J. C 83, 304 (2023). https://doi.org/10.1140/epjc/s10052-023-11435-9.
	 
**CRANN** contains trained artificial neural networks to generate synthetic cosmic chronometers, JLA Type Ia supernovae (distance modulus) and 
<img src="https://render.githubusercontent.com/render/math?math=f_{\sigma8}"> data.

For the training details and development, visit the following repository: https://github.com/igomezv/neuralCosmoReconstruction .

Requiriements:

- numpy
- sklearn
- scipy
- pandas
- matplotlib
- seaborn
- tensorflow==2.6.0
- keras==2.6.0
- astroNN
- h5py==2.9.0
