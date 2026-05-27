# ChessClassification

Synthetic-to-Real Generalization for Chessboard square and board-state Classification

## Table of Contents

- [Collaborators](#collaborators)
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Inference](#inference)
    - [Developer](#developer)
    - [End User](#end-user)
- [Technical](#technical-dataset-approach-and-results)
- [Skills](#skills)
- [Academic Citation](#academic-citation)

## Collaborators

This project was completed by:

- [Jonathan Zmora](https://github.com/JonathanZmora)
- [Lior Vinman](https://github.com/liorvi35)

We are first-year M.Sc. students in the Department of Computer and Information Science at
[Ben Gurion University](https://www.bgu.ac.il/en/).

## Introduction

This repository presents our Final Project for the Introduction to Deep Learning course at Ben Gurion University.

The project focuses on recovering the full state of a chess game from a single RGB image of a physical chessboard,
namely, a real-world image captured during an actual game between two players.

More specifically, the task is to classify each of the 64 squares on the board, reconstruct the complete board
configuration, and represent it in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
format, which uniquely describes a given board state.

Because labeled real-world data is both limited and costly to collect, clean, and annotate,
this project explores the use of mostly [synthetic data](https://en.wikipedia.org/wiki/Synthetic_data)
generated through a digital rendering pipeline, together with learning strategies designed to
improve generalization from synthetic images to real photographs.

The course is taught by [Prof. Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/index.htm), with the course TA, Mr. Roy Amoyal.

## Project Overview

The goal of this project is to recover the full state of a chess game from an image of a physical chessboard.
Given a single input image, the model identifies the content of the board and reconstructs the corresponding
position in FEN format.

A central challenge in this task is the significant gap between the synthetic training data and the real test data.
To address this challenge, the project relies on large scale synthetic data generation and examines how well
models trained on synthetic data transfer to real world photographs.

The project includes data preparation scripts, model training, evaluation on both synthetic and real images,
and experiments with ResNet18, ConvNeXt, Transformer-based board context, and an ILP solver that enforces
valid chessboard constraints during inference.

Overall, the project develops and studies a complete pipeline for chessboard state recognition from physical
chessboard images.

## Google Drive

All Project's static data sources are found in ours academic
[Google Drive](https://drive.google.com/drive/u/0/folders/1JJbqjPhAtJAhZVrHCIJQ19dPz8wSFkLP).
There you could find 3 main building block of our research and project:

* Naive data - at `naive_synthetic_data/`, which is the synthetic data that has been generated using the original
Blender script, we've received: those are fully ideal black-and-white synthetic images, with perfect lighting and
sharp resolution. 

* Quality data - at `quality_synthetic_data/`, which is synthetic data that has been generated via our improved
generation pipeline: improved resolution, coloring, padding, adding noise - which here to reduce the gap between
the synthetic dataset and the real-world dataset.

**all data that is found on the Google Drive, is divided into 3: validation, train and test.
And, is compressed into a `zip` format.**

* Pre Trained Models - at `models/`, there you could find all the models we've created during our performance research,
all of them are available there.

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

The project should now be ready to use.

## Usage

Here we'll discuss the overall usage of the project: retrain and infer.

### Train

To train our models, we created a script `trainer.py` which is a simple command-line tool, that does a setup, train,
validation, test and dumps a new model of same architecture that is trained over custom datasets.

**The script is found in the root directory of this repository**

_Important Note_:

Every model requires a setup of its own - which requires a script support and maintenance, this why this script allows
to re-train only the models that we thought of it being interesting to retreain them on another data, which are the
following:

1. convnext_zero_shot
2. convnext_transformer 
3. convnext_fine_tuned_final_stage

**available flags**:

* `--train` - path to a directory with the train dataset
* `--val` - path to a directory with the validation dataset 
* `--test` - path to a directory with the test dataset
* `--model-name` - one of: "convnext_zero_shot", "convnext_transformer", "convnext_fine_tuned_final_stage"
* `--model-path` - path to the `pth` model file
* `--save-path` - destination path, where save the new re-treined model
* `--lr` - learning rate
* `--epochs` - amount of train epochs
* `--batch` - amount of samples per batch
* `--padding` - crop padding for each cell in the board
* `--schedualer` - enable a CosineAnnealingLR scheduler during training

Here is an example running the training script, this example could be used directly on a Linux shell environment:

```shell
python trainer.py --train=data/train/synthetic --val=data/validation/synthetic \ 
  --test=data/test/real --model-name=convnext_transformer --model-path=models/convnext_transformer.pth \
  --save-path=trained_model.pth --epochs=2 --lr=0.0001 --scheduler
```

### Inference

Here we'll show a few ways of running inference on one of our pre-ready models, which are discussed in the article,
they could be found [here](https://drive.google.com/drive/u/0/folders/1nLEzm4LjWIXToCoKd1IA6LpzQ0f2tguT).

The idea is we show 2 ways of running the predictions, the one is for develops, so you could integrate our models into
your own code base/architecture and the second is if you only want to use our strongest model.

#### Developer

After the project is successfully cloned and installed (see [Prerequisites](#Prerequisites) and [Installation](#installation)),
you may want to use the `predict_board()` function, which returns a prediction for a single chessboard picture.

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

import numpy as np
import torch

from src.network.inference import predict_board

# setting a model path as env
os.environ["PATH_TO_MODEL"] = "/home/user/Documents/model.pth"

# getting an image, as numpy array
my_chessboard_image: np.ndarray = np.ndarray(...)

# making an inference of the image, via the configured model
board_prediction: torch.Tensor = predict_board(my_chessboard_image)
```

#### End-User

It is possible to run image predictions using our
[web app](https://jonathanzmora.github.io/ChessClassification/app.html).

First of all, press `Ping Backend` to ensure the BE virtual machine is up and working,
then you can press `Test me!` to use a random sampled image from a small dataset,
or press `Choose image` and then `Classify board` to upload and predict image of your own. 

## Technical: Dataset, Approach and Results

For a more detailed discussion, please refer to: [Project's academic web-page](https://jonathanzmora.github.io/ChessClassification/)

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
