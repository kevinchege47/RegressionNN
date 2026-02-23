# Python Project

## Description

This repository contains a Python script demonstrating neural network training using PyTorch for regression on synthetic linear data with added noise. It implements early stopping to prevent overfitting, model checkpointing, evaluation, and data visualization.

## Files

- `main.py`: Trains a simple neural network to fit a linear relationship (y = 3x + 1) with noise, using early stopping, model saving, and plotting the results.

- `best_model.pt`: Saved PyTorch model state from `main.py`.

- `best_housing.pt`: Additional saved model file (from a previous script).

## Requirements

- Python 3.x
- PyTorch
- matplotlib
- numpy

## Installation

Install the required packages using pip:

```
pip install torch matplotlib numpy
```

## Usage

Run the script:

- `python main.py`: Trains the model, saves the best model, evaluates it, and displays a plot of the data and fitted line (requires a display for matplotlib).

## Notes

- The script uses a random seed for reproducibility.
- Early stopping is implemented to halt training when validation loss stops improving.
- The model is saved to `best_model.pt` for later use.
