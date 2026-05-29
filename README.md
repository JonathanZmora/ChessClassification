# ChessClassification

Synthetic-to-Real Generalization for Chessboard square and board-state Classification

## Table of Contents

- [Collaborators](#collaborators)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Inference](#inference)
    - [Developer](#developer)
    - [End User](#end-user)
- [Skills](#skills)
- [Academic Citation](#academic-citation)

## Collaborators

This project was completed by:

- [Jonathan Zmora](https://github.com/JonathanZmora)
- [Lior Vinman](https://github.com/liorvi35)

We are first-year M.Sc. students in the Stein faculty of Computer and Information Science at
[Ben Gurion University](https://www.bgu.ac.il/en/).

## Project Overview

This repository presents our final project for the Introduction to Deep Learning course, taught by [Prof. Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/index.htm) and Mr. Roy Amoyal.

The project focuses on recovering the full state of a chess game from a single RGB image of a physical chessboard,
namely, a real-world image captured during an actual game between two players.

More specifically, the task is to classify each of the 64 squares on the board, reconstruct the complete board
configuration, and represent it in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
format, which uniquely describes a given board state.

Because labeled real-world data is both limited and costly to collect, clean, and annotate,
this project explores the use of mostly synthetic data generated through a digital rendering
pipeline, together with learning strategies designed to improve generalization from synthetic 
images to real photographs.

A central challenge in this task is the significant gap between the synthetic training data and the real test data.
To address this challenge, the project relies on quality synthetic data generation and examines how well
models trained on synthetic data transfer to real world photographs.

The project includes data generation and pre-processing, model training and evaluation, 
experiments with different neural network architectures, and an Integer Linear Programming
solver that enforces valid chessboard constraints during inference.
All models were traing using the PyTorch framework. 

Overall, the project develops and studies a complete pipeline for chessboard state recognition from physical
chessboard images.

For a more detailed discussion, please refer to: [Project's academic web-page](https://jonathanzmora.github.io/ChessClassification/)

## Google Drive

All Project's static data sources can be found in our academic
[Google Drive](https://drive.google.com/drive/u/0/folders/1JJbqjPhAtJAhZVrHCIJQ19dPz8wSFkLP).
There you could find all of the data we used for training, validation, and testing, as well as 
the weights for all of the models we trained during our experiments.
Access to the different contents is as follows:

* models/ - This directory contains `.pth` files with weights for all the models we've trained during our experiments.

* naive_synthetic_data/ - This directory contains the synthetic data that we generated using the original
Blender generation script we received with the project insturctions.
Those are ideal black-and-white synthetic images which we used for our first baseline experiments.

* quality_synthetic_data/ - This directory contains the synthetic data that we generated using our improved
custom generation pipeline.

* Inside each of the above directories, you will find train, validation, and test directories, each containing
  the data for the respective split. The data itself is compressed into a `zip` format inside the dataset.zip file.
  To access the data, you must download the file and unzip it. All datasets are organized as follows:

  dataset/
  
  	└─ images/

  	└─ gt.csv

	The gt.csv file contains 3 columns (and more for synthetic data):
	1. image_name
	2. FEN string corresponding to the image
	3. View specification (black/white)

## Prerequisites

Before installing and running this project, ensure that the following are installed on your system:

- Python 3.10 or newer
- pip (Python package manager)
- Git

After installation, verify that `python`, `pip`, and `git` are available as command-line tools in your environment, via:

**Windows & Linux**
```shell
python --version
pip --version
git --version
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/JonathanZmora/ChessClassification.git
cd ChessClassification
```

### Create a Virtual Environment (Recommended)

It is recommended to use a virtual environment to keep project dependencies isolated.

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux**
```bash
source .venv/bin/activate
```

### Install Dependencies

Install the required packages with:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

_Important Note_:

Because PyTorch depends on your specific hardware (CPU, Mac, or specific NVIDIA GPUs), 
please visit the [Official PyTorch Get Started Page](https://pytorch.org/get-started/locally/)
and install the version appropriate for your system if you need to.
A specific torch and torchvision version is not specified in the requirements.txt file.
For example, for NVIDIA GeForce GTX 1080 Ti GPU you can use: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

The project should now be ready to use.

## Usage

### Train

To train our best performing zero-shot and fine-tuned models, we created a `trainer.py` script, which is a simple command-line tool
that takes care of the data setup, model initialization, training and evaluation both with and without using the ILP solver.
The new trained model will be saved at a path of your choosing.
Full usage explanation can be found in the DocString inside the `trainer.py` file.
In order to re-create our exact experiment configurations, here are the full commands you should use (swapping only the paths to the correct ones):

* For our best performing zero-shot model:

	```bash
	python trainer.py \
 	--train data/train/synthetic \
 	--val data/validation/synthetic \
 	--test data/test/real \
 	--model-name convnext_transformer \
 	--model-path models/convnext_zero_shot.pth \
 	--save-path models/trained_model.pth --epochs 2 \
	--lr 0.0001 \
 	--scheduler
 	```

* For our best performing fine-tuned model:

	```bash
	python trainer.py \
 	--train data/train/real \
 	--val data/validation/real \
 	--test data/test/real \
 	--model-name convnext_fine_tuned_final_stage \
 	--model-path models/convnext_zero_shot.pth \
 	--save-path models/trained_model.pth \
 	--epochs 15 \
	--lr 0.001 \
 	--scheduler
	```
 
**The script resides in the root directory of this repository**

_Important Notes_:

1. You are only allowed to re-train our best performing zero-shot model and our best performing fine-tuned model,
which are named  "convnext_transformer" and "convnext_fine_tuned_final_stage" respectively.
These are the only names that the script allows as --model-name argument values.
2. Note that the model weights file path passed to the script should be the `convnext_zero_shot.pth`
file found in the google drive models directory.

### Inference

Here we'll show 2 methods of running inference on one of our trained models, which are discussed in the article.

#### Developer

After the project is successfully cloned and installed (see [Prerequisites](#Prerequisites) and [Installation](#installation)),
you may want to use the `predict_board()` function, which returns a prediction for a single chessboard image.

Before using it, you need to set an env variable, with the model's path. This could be done via Python:
```python
import os
os.environ["PATH_TO_MODEL"] = "/path/to/model.pth"
```

or via shell - here are few examples for common shell environments, both on Windows and Linux,

**Windows CMD**:
```shell
set PATH_TO_MODEL="/path/to/model.pth"
```

**Windows PowerShell**:
```shell
$env:PATH_TO_MODEL = "/path/to/model.pth"
```

**Linux bash**:
```shell
export PATH_TO_MODEL="/path/to/model.pth"
```

_Only after exporting such environment variable_, you can proceed and import the method and use it.
See this E2E example for using our SDK:
```python
import os
import cv2
import numpy as np
import torch

from src.network.inference import predict_board

# setting a model path as env
os.environ["PATH_TO_MODEL"] = "/home/user/ChessClassification/models/model.pth"

# getting an image as numpy array
image_path: str = "/home/user/ChessClassification/data/test/real/images/image.jpg"
my_chessboard_image: np.ndarray = cv2.imread(image_path)

# making predictions on the image, via the configured model
board_prediction: torch.Tensor = predict_board(my_chessboard_image)
```

#### End-User

It is possible to run image predictions using our
[web app](https://jonathanzmora.github.io/ChessClassification/app.html).

First of all, press `Ping Backend` to ensure the virtual machine is up and working,
then you can press `Test me!` to use a random sampled image from a small dataset,
or press `Choose image` and then `Classify Board` to upload and classify an image of your own. 

## Skills

This project demonstrates practical work in:

<div align="center">
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/git.png" alt="Git" title="Git"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/github.png" alt="GitHub" title="GitHub"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/linux.png" alt="Linux" title="Linux"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/pycharm.png" alt="PyCharm" title="PyCharm"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/jupyter_notebook.png" alt="Jupyter Notebook" title="Jupyter Notebook"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/python.png" alt="Python" title="Python"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/latex.png" alt="LaTeX" title="LaTeX"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/html.png" alt="HTML" title="HTML"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/css.png" alt="CSS" title="CSS"/></code>
	<code><img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/javascript.png" alt="JavaScript" title="JavaScript"/></code>
</div>

## Academic Citation

If you use this repository in academic research, please cite:

```bibtex
@misc{zmora_vinman_synthetic_2026,
  title        = {Synthetic Is All You Need?},
  author       = {Jonathan Zmora and Lior Vinman},
  institution  = {Ben-Gurion University of the Negev},
  year         = {2026},
  note         = {Introduction to Deep Learning project},
  url          = {https://github.com/JonathanZmora/ChessClassification}
}
```

If this work contributes to your academic research, we kindly request that you cite it.
Citation helps support our academic development and acknowledges the effort invested in this project!
