 
[<img src="https://img.shields.io/badge/astro--ph.CO-%20%09arXiv%3A2104.00595-red.svg">](https://arxiv.org/abs/2104.00595)
      
# Cosmological Reconstructions with Artificial Neural Networks (CRANN)

*This repository is under construction. When the paper is published, it will be completed*.

In arxiv_notebooks folder, there are the notebooks that generates the figures of the paper.

If you use this code please cite our preprint:
	
	Gómez-Vargas, Isidro and Medel Esquivel, Ricardo and García-Salcedo, Ricardo and Vazquez, J. Alberto, Neural Network Reconstructions for the Hubble Parameter, Growth Rate and Distance Modulus. DARK-D-21-00528, Available at SSRN: https://ssrn.com/abstract=3990646 or http://dx.doi.org/10.2139/ssrn.3990646 .
	 
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
