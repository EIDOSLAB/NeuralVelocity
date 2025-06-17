# 🧠 [IJCNN 2025] NeVe: Neural Velocity for hyperparameter tuning

[![Docker Ready](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)
[![GPU Support](https://img.shields.io/badge/GPU-Supported-green?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-388/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

This repository contains the official implementation of the paper:
> **Neural Velocity for hyperparameter tuning**  
> *Gianluca Dalmasso, et al.*  
> IJCNN 2025  
> 📄 [arXiv / DOI link here]

---

![Teaser](assets/teaser.png)

## 📂 Project Structure
```bash
NeuralVelocity/
├── 📁 assets/                   # Teaser images, figures, etc.
├── 📁 src/
│   ├── 📁 dataloaders/          # CIFAR, ImageNet loaders
│   ├── 📁 labelwave/            # Competing method: LabelWave
│   ├── 📁 ultimate_optimizer/   # Competing method: Ultimate Optimizer
│   ├── 📁 neve/                 # 💡 Core method: Neural Velocity
│   ├── 📁 models/               # Model architectures (e.g. CIFAR ResNets, INet ResNets, ...)
│   ├── 📁 optimizers/           # Optimizers
│   ├── 📁 schedulers/           # LR schedulers
│   ├── 📁 swin_transformer/     # Swin Transformer model architecture
│   ├── arguments.py                          # CLI args and config parser
│   ├── classification.py                     # Training pipeline (base)
│   ├── classification_labelwave.py           # For LabelWave experiments
│   ├── classification_ultimate_optimizer.py  # For Ultimate Optimizer experiments
│   └── utils.py                              # Utility functions
├── Dockerfile                   # Default Docker container
├── Dockerfile.python            # Base Python environment
├── Dockerfile.sweep             # Sweep setup (e.g. for tuning)
├── LICENSE                      # GNU GPLv3 license
├── README.md                    # Project overview
├── build.sh                     # Build script (e.g. for Docker or sweep)
├── requirements.txt             # Python dependencies
└── setup.py                     # Install package for pip
```

---

## 🚀 Getting Started
You can run this project either using a Python virtual environment or a Docker container.

#### ✅ Clone the repository
```bash
git clone https://github.com/EIDOSLAB/NeuralVelocity.git
cd NeuralVelocity
```

### 🧪 Option A — Run with virtual environment (recommended for development)

#### 📦 Create virtual environment & install dependencies
> This project was developed and tested with Python 3.8.8 — we recommend using the same version for full compatibility and reproducibility.
```bash
# 1. Install Python 3.8.8 (only once)
pyenv install 3.8.8

# 2. Create virtual environment
pyenv virtualenv 3.8.8 neve

# 3. Activate the environment
pyenv activate neve

# 4. Install dependencies
pip install -r requirements.txt
```

#### 🚀 Run training
```bash
cd src
python classification.py
```

### 🐳 Option B — Run with Docker
You can also use Docker for full environment reproducibility.

#### 🏗️ Build Docker images and push to remote registry
The `build.sh` script automates the build of all Docker images and pushes them to the configured remote Docker registry.

Before running, make sure to edit `build.sh` to set your remote registry URL and credentials if needed.

Run:
```bash
bash build.sh
```
This will build the following Docker images:
- `neve:base` (default container for training and experiments)
- `neve:python` (base Python environment)
- `neve:sweep` (for hyperparameter sweep experiments)
    
#### 🚀 Run training inside the container
```bash
docker run --rm -it \
  --gpus all \                   # Optional: remove if no GPU
  neve:python classification.py  # Optional: Optional parameters...
```
> 💡 Note: you may need to adjust volume mounting (-v) depending on your OS and Docker setup.

---

## 📊 Datasets
Tested datasets:
 - [CIFAR10, and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [Imagenet-100](https://www.image-net.org/challenges/LSVRC/2012/) (must be downloaded separately and prepared in the standard folder format.)

---

## 🪪 License
This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](./LICENSE) file for details.

➡️ You are free to use, modify, and distribute this code under the same license terms.  
Any derivative work must also be distributed under the GNU GPL.

---

## 🙌 Acknowledgments
This research was developed at the University of Turin (UniTO), within the [EIDOS Lab](https://www.di.unito.it/~eidos/), and Télécom Paris.

We thank the members of both institutions for the insightful discussions and support during the development of this work.


---

## 📜 Citation
If you use this repository or find our work helpful, please cite:
```bibtex
@misc{dalmasso2025neve,
  title        = {Neural Velocity for Hyperparameter Tuning},
  author       = {Gianluca Dalmasso and Others},
  year         = {2025},
  howpublished = {\url{https://arxiv.org/abs/xxxx.xxxxx}},
  note         = {Accepted at IJCNN 2025. Official citation will be updated upon publication.}
}
```

---

## 📫 Contact
For questions or collaborations, feel free to reach out:
- 📧 gianluca.dalmasso@unito.it
- 🐙 GitHub Issues for bugs or feature requests
