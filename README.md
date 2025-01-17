# From Cueva to Campagner

## `diary.ipynb`

Jupyter Notebook tracking progress made.

## `Build.py`

Contains function `build`, which creates, trains, tests, and saves a model from start to finish. The model depends on the `Config` object supplied - see `Config.py` 

To run default build of 'ego0D' (egocentric zero-dimensional) with the original Cueva et al. (2020) settings, in this direcory run:

`python Build.py`

Modify default settings with flags `-<setting name> <setting value>` (see `default_params` in `Config.py` for full list of settings and default values), e.g. `python Build.py -task allo1D -n_neurons 500`

## `Config.py`

Contains `Config` object for handling model build configurations, and list of default build parameters `default_params`.

## `Data.py`

Contains dataset classes used for each task (i.e. where task data is generated). Original task from Cueve et al., 2020 is `ZeroDimEgocentricDataset`; ultimate goal will be to train on `TwoDimAllocentricDataset`.

## `RNN.py`

Class implementing Continuous-Time Recurrent Neural Network functionality, in accordance with task-specific structure (defined with a `Config` object).

## `Train.py`

Contains function for conducting a set of training iterations. Is used by `build`.

## `Test.py`

Contains functions for conducting a series of tests on a given model build, and generating plots of the results. Most plots are equivalent to those included in Cueva et al., 2020. Is used by `build`.

## `HF.py`

Optimiser class for implementing Hessian-Free learning algorithm (Martens & Sutskever, 2012); code adapted from https://github.com/fmeirinhos/pytorch-hessianfree.

## trained-models/

A directory containing the results of model builds (i.e. the outputs of various instances of `build`). Sub-directories here (prefixed with `<time>-task:<task>`) are build instances (i.e., specific `Config` settings, with important settings included in the directory name).

Sub-sub-directories within each build (prefixed with `threshold:`) are checkpoints in the model build when their loss on the training dataset fell below the specified threshold.

These checkpoint directories contain the checkpoint itself `epoch:<>-loss:<>.pt` and the plots generated in the course of testing.