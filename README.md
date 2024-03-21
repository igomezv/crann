[<img src="https://img.shields.io/badge/astro--ph.CO-arXiv%3A2104.00595-red.svg">](https://arxiv.org/abs/2104.00595)

# Cosmological Reconstructions with Artificial Neural Networks (CRANN)

**CRANN** is an open-source project designed to facilitate cosmological reconstructions using Artificial Neural Networks. This repository hosts the neural networks trained to generate synthetic cosmic chronometers, JLA Type Ia supernovae (distance modulus), and $f_{\sigma8}$ data, based on the findings and methodologies described in our paper:

> Gómez-Vargas, I., Medel-Esquivel, R., García-Salcedo, R. et al. Neural network reconstructions for the Hubble parameter, growth rate, and distance modulus. Eur. Phys. J. C 83, 304 (2023). [https://doi.org/10.1140/epjc/s10052-023-11435-9](https://doi.org/10.1140/epjc/s10052-023-11435-9).

## Repository Contents

- **arxiv_notebooks/**: Contains Jupyter notebooks that reproduce the figures presented in the paper. These notebooks provide insights into the data generation process and the neural network training details.

## Getting Started

To use the CRANN models or to contribute to the project, you will need to install several dependencies. Ensure you have the following Python packages installed:

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

You can install these packages using `pip` by running:

```bash
pip install numpy sklearn scipy pandas matplotlib seaborn tensorflow==2.6.0 keras==2.6.0 astroNN h5py==2.9.0
```

## Usage
Details about the training of the models can be found in the **arxiv_notebooks/** directory. 

## Citing This Work

If you use CRANN in your research, please cite our paper to acknowledge the work that has gone into developing this resource:

```bibtex
@article{GomezVargas2023,
  title={Neural network reconstructions for the Hubble parameter, growth rate and distance modulus},
  author={Gómez-Vargas, I. and Medel-Esquivel, R. and García-Salcedo, R. and others},
  journal={Eur. Phys. J. C},
  volume={83},
  pages={304},
  year={2023},
  publisher={Springer}
}
```

## Additional Resources

For more details on the training process and development of these models, please refer to our related repository: [neuralCosmoReconstruction](https://github.com/igomezv/neuralCosmoReconstruction).

## Note

This repository is currently under construction. We welcome contributions and suggestions to improve the project.
