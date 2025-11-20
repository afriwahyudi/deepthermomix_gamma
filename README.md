# DeepThermoMix

## Project Overview

DeepThermoMix is a physics-informed machine learning model capable of predicting multicomponent activity coefficients.

### Key Architectural Components
DTMPNN is an end-to-end framework that processes molecular structure into activity coefficient:
- **Message Passing Neural Networks (MPNNs)** to encode SMILES into continuous component embedding space
- **Deep Thermodynamic Mixing Layer (DTM)** to encode component embedding space into mixture embeddings

### Core capabilities:
|Capability|Details|
|-------------------------------|--------------------------------------------------------------------------------|
| **Component agnostic**        | Accepts arbitrary N >= 2 components in solvent mixture                         |
| **Permutation Invariant**     | The prediction is invariant to the order of component appearance               |
| **Learnable Mixing Rule**     | Employs a learnable non-linear mixing rule derived directly from the DTM layer |
| **Gibbs–Duhem Trained**       | Imposes thermodynamic consistency in solvent representation                    |

## Project Environment
### Hardware specification
    - CPU: Intel(R) Core(TM) i9-13900KF
    - GPU: NVIDIA RTX A4000 16 GB VRAM
    - RAM: 128 GB DDR5

### Installation Step
#### Step 1. Core requirements
	1. Install NVIDIA driver
	2. Install CUDA Toolkit (cu130)
	3. Install cuDNN
#### Step 2. Library setup
	1. conda create --name gamma-dtmpnn python=3.11
	2. conda activate gamma-dtmpnn
	3. pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
		- python -c "import torch; print(torch.cuda.is_available())"
		- pip show torch
	4. pip install torch_geometric
	5. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
		- pip show torch-geometric
	6. conda install pandas matplotlib seaborn scikit-learn networkx rdkit jupyter ipykernel -c conda-forge -y
	7. pip install captum plotly kaleido optuna optuna-dashboard papermill
#### Step 3. Confirm the installation
    1. Go to project root
    2. Run check_env.py
	3. The following package information serves as a reference to ensure full environment reproducibility
---
### Project information

#### Library Versions
| Package              | Version / Status                |
|----------------------|---------------------------------|
| pandas               | 2.3.3                           |
| numpy                | 1.26.4                          |
| matplotlib           | 3.10.7                          |
| seaborn              | 0.13.2                          |
| sklearn              | 1.7.2                           |
| networkx             | 3.5                             |
| rdkit                | 2025.09.1                       |
| joblib               | 1.5.2                           |
| tqdm                 | 4.67.1                          |
| torch                | 2.8.0+cu129                     |
| torchvision          | 0.23.0+cu129                    |
| torchaudio           | 2.8.0+cu129                     |
| torch_geometric      | 2.7.0                           |
| pyg_lib              | 0.5.0+pt28cu129                 |
| torch_scatter        | 2.1.2+pt28cu129                 |
| torch_sparse         | 0.6.18+pt28cu129                |
| torch_cluster        | 1.6.3+pt28cu129                 |
| torch_spline_conv    | 1.2.2+pt28cu129                 |
| optuna               | 4.5.0                           |
| optuna_dashboard     | 0.19.0                          |
| captum               | 0.8.0                           |
| plotly               | 6.3.1                           |
| kaleido              | Installed (no version attribute)|
| papermill            | 2.6.0                           |
| jupyter              | Installed (no version attribute)|
| ipykernel            | 7.1.0                           |

#### PyTorch & CUDA Info

| Property             | Value             |
|----------------------|-------------------|
| Torch version        | 2.8.0+cu129       |
| CUDA available       | True              |
| CUDA version         | 12.9              |
| cuDNN version        | 91002             |
| GPU device count     | 1                 |
| Current GPU device   | NVIDIA RTX A4000  |
---