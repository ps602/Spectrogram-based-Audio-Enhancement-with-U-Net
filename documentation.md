# Spectrogram-based Audio Enhancement with U-Net

This repository contains the implementation of U-Net (baseline) and U-Net (with Time-Shift Convolution) for high-fideltiy audio reconstruction using Spectrograms. Users can easily switch between different models and choose between training and evaluation modes using command-line arguments.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Supported Models](#supported-models)
- [Contributing](#contributing)

## Installation

To set up the environment to use this framework, follow these steps:

```bash
# Clone the repository
git clone https://your-repo-url.git
cd your-repo-directory

# (Optional) Set up a virtual environment
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```
## Usage

You can run the framework in training or evaluation mode for a specified model using command-line arguments.

# Training
To train a model, use the following command:

```bash
python main.py --method train --model <MODEL_NAME>
```

# Evaluation
To evaluate a model on the test set, use:

```bash
python main.py --mode eval --model <MODEL_NAME>
```

Replace <MODEL_NAME> with "unet" or "unet_tsc" for baseline U-Net or U-Net with Time-shift Convolution respectively. 

## Repostiory Structure
├── main.py                # Main script to run training or evaluation
├── models.py              # Defines the supported models
├── results                # Spectrogram/Waveform images of input and output
├── data                   # Training Data and Evaluation Data
├── requirements.txt       # Lists dependencies for the project
└── README.md              # Documentation for the project

# Supported Models

This framework currently supports the following models:

Model1: U-Net Baseline Architecture with Conv2D Layers
Model2: U-Net + Time-Shift Convolution architecture

For detailed information on each model, please refer to the models.py file.



