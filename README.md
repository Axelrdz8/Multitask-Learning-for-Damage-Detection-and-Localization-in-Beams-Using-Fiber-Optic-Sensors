# Repository workflow

This repository documents an experimental workflow for damage detection and localization in beams instrumented with a optical fiber sensor.

<img width="715" height="365" alt="sensor" src="https://github.com/user-attachments/assets/fd49b90a-00e4-46b3-ba1d-a17ebdc164dd" />
<img width="514" height="418" alt="diagram" src="https://github.com/user-attachments/assets/2cb60c1f-38fb-4676-a11f-4e84510cd0c4" />


## Workflow overview

- `beam_vibration_dataset` contains the raw measurements in CSV format, separated by structural condition.
- `notebooks/analysis-codes` is used for exploratory inspection of signals and FFT comparisons.
- `notebooks/build-dataset` transforms raw captures into a tabular feature dataset.
- `notebooks/single task models-*` trains baseline models per task.
- `notebooks/multitask-model` contains the final multitask model and its export.

## Root folder

- `beam_vibration_dataset`: experimental base of the project.
- `notebooks`: implementation of the analysis workflow, dataset construction, and training.

## `beam_vibration_dataset` folder

This folder stores the original CSV files from vibration acquisitions.

<img width="743" height="402" alt="dataset" src="https://github.com/user-attachments/assets/acb7149a-0c0f-4309-a925-0dfd79f67df3" />

In total, the raw folder contains 1900 CSV files organized by damage mechanism:

- `healthy`:
  contains 250 reference captures without damage. Directories `ALL0000` to `ALL0099` and their copies represent acquisition series used to describe the baseline behavior of the beam.
- `crack`:
  contains 450 CSV files of isolated crack damage. It is divided into three severities (`1-slight`, `2-moderate`, `3-severe`) and within each one there are positions such as `1.5cm`, `2.5cm`, ..., `6.5cm` along a `10cm` beam.
- `wear`:
  contains 450 CSV files of isolated wear damage. The structure is equivalent to `crack`, but focused on the wear mechanism.
- `simultaneous`:
  contains 750 CSV files of combined damage. Its subfolders encode mixed scenarios such as `severecrack_slightwear`, `severewear_moderatecrack`, etc. In this part of the project, both regression outputs are activated simultaneously.

## `notebooks` folder

This folder concentrates the project logic.

### `analysis-codes` subfolder

Used for exploration and prior validation of signals.

- `fft_comparison.ipynb`:
  notebook aimed at comparing the spectral content of healthy measurements and visualizing frequency peaks.
- `fft_comparison2.ipynb`:
  second variant of FFT analysis, likely used to refine plots and contrasts between readings.
- `healthy_frequency_table.csv`:
  summary table with healthy frequencies, used as a reference for nominal behavior.
- `healthy_beam_lectures`:
  organized subset for detailed inspection. Regroups measurements to facilitate comparison of healthy conditions by configuration or position.

### `build-dataset` subfolder

- `build-dataset-csv.py`:
  main script of the repository. Traverses `beam_vibration_dataset`, extracts metadata from folder names, computes FFT, obtains the top 5 peaks, and generates spectral features such as `FC_Hz`, `FRMS_Hz`, `FRVF_Hz`, `m0`, `m1`, and `m2`.
- `dataset_beam.csv`:
  consolidated tabular output for training.

The script also generates the multitask labels of the project:

- `target_class`: structural state classification.
- `distance_crack_cm`: crack localization.
- `distance_wear_cm`: wear localization.
- `crack_mask` and `wear_mask`: indicate when each regression applies.

With the current configuration, the final CSV has 3600 records and 12 balanced classes of 300 samples each. The balance appears because several captures are split into two halves before feature extraction, especially for `crack`, `wear`, and `simultaneous`.

### `single task models-classification` subfolder

Contains the classification baseline as an isolated problem.

- `single_task_classif.ipynb`:
  training and evaluation notebook.
- `dataset_beam.csv`:
  local copy of the dataset to keep the notebook self-contained.
- `cls_grid_models`:
  search results and training/testing metrics.

This folder allows measuring how much can be solved using only the classification output, without sharing representations with localization tasks.

### `single task models-regression-crack` subfolder

Implements crack distance regression as an independent task.

- `single_task_regress_crack.ipynb`:
  main training notebook.
- `crack_reg_grid_models`:
  metrics and comparative results.

Here, models such as `KNN`, `SVR`, `MLP`, `DecisionTree`, `Ridge`, `Lasso`, and `ElasticNet` are compared.

### `single task models-regression-wear` subfolder

Replicates the previous scheme, but for wear localization.

- `single_task_regress_wear.ipynb`:
  main notebook for wear.
- `wear_reg_grid_models`:
  quantitative results of the evaluated models.

The idea is to decouple wear localization from general classification and measure how far a specialized model can go without multitask learning.

### `multitask-model` subfolder

<img width="684" height="514" alt="best_model" src="https://github.com/user-attachments/assets/d81ea7c6-2d5f-4fd5-9a1c-d4582e0d51ca" />

- `multitask-model.ipynb`:
  notebook that trains a multitask neural network with Keras/TensorFlow.
- `dataset_beam.csv`:
  copy of the consolidated dataset.
- `export_beam_model_20260318_171501`:
  folder with exported artifacts.

Inside the exported folder:

- `model.keras`: serialized final model.
- `best_hparams.json`: best hyperparameters found.
- `features.json`: set of 10 selected features.
- `meta.json`: model metadata, including `12` classes and loading compatibility with Keras 3.
- `training_history.csv`: training and validation history.

This subfolder integrates the entire previous workflow: it takes the feature dataset and produces a network with shared representation to simultaneously predict class, crack distance, and wear distance.

## Full project workflow

The repository can be understood as a six-step pipeline:

- raw signal acquisition in `beam_vibration_dataset`
- exploratory inspection with FFT in `analysis-codes`
- feature engineering in `build-dataset`
- generation of the balanced tabular dataset
- training of single-task baselines
- training and export of the final multitask model
