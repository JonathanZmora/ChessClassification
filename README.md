# ChessClassification

Synthetic-to-Real Generalization for Chessboard square and board-state Classification

## Table of Contents

- [Collaborators](#collaborators)
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
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
By nature, the synthetic images are highly controlled: they are black and white, idealized, noiseless, and lack many
of the visual characteristics found in real photographs, such as color variation, lighting changes, blur, shadows,
and background clutter.

To address this challenge, the project relies on large scale synthetic data generation to produce labeled training
examples in a setting where real annotated data is limited and expensive to obtain. We then examine how effectively
models trained on synthetic data transfer to real world photographs, and how different training and evaluation choices
influence that transfer.

The project includes data preparation scripts, model training, evaluation on both synthetic and real images,
and a series of experiments analyzing how synthetic training data can improve real world chessboard recognition.

Overall, the project develops and studies a complete pipeline for chessboard state recognition
using **ResNet18** and **ConvNeXt** models.

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

## Dataset

## Approach

## Results

Our final model achieved 97.3% accuracy on the real-world test dataset.

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
@misc{zmora_vinman_chessclassification,
  title  = {Synthetic Is All You Need?},
  author = {Jonathan Zmora and Lior Vinman},
  year   = {2026}
}
```
