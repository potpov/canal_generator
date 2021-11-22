# Improving Segmentation of the Inferior Alveolar Nerve through Deep Label Propagation - Canal generator

This repository contains the material from the paper "Improving Segmentation of the Inferior Alveolar Nerve through Deep Label Propagation".

In particular, this repo is dedicated to the deep expansion network.
our new 3D dense dataset can be downloaded [here](#linkhere).

## Usage
you can run the experiments as follow:
```
usage: main.py [--base_config path]

optional arguments:
  --base_config         path to your config.yaml for this experiment
  --verbose             redirect stream to std out instead of using a log file in the yaml directory
  --competitor          load training data as circle expansion instead of dense annotations
  --additional_dataset  load the additional patients
  --test                skip the training and load best weights for the experiment (no needs to update your yaml file)
  --skip_dump           if this flag is set the network does not dump prediction volumes on the final test
  --reload              load the last weights and continue the training (no needs to update your yaml file)
```

Optional Arguments for the distributed data parallel. note, you must configure your slurm sbatch file to use it

```
--local_rank              
--dist-backend              
--dist-url
```

## YAML config example
Here is an example of a yaml file to use as base_config. The following is the yaml file used in the experiment which obtained the best values 

```yaml
data-loader:
  augmentations_file: /augmentations_files/myaug.yaml
  background_suppression: 0
  batch_size: 2
  file_path: /datasets/maxillo/DENSE
  labels:
    BACKGROUND: 0
    INSIDE: 1
  mean: 0.08435
  num_workers: 4
  patch_shape:
  - 120
  - 120
  - 120
  resize_shape:
  - 168
  - 280
  - 360
  sparse_path: /datasets/maxillo/SPARSE
  split_filepath: /splits/generator_training.json
  std: 0.17885
  volumes_max: 2100
  volumes_min: 0
  weights:
  - 0.000703
  - 0.999
loss:
  name: Jaccard
lr_scheduler:
  name: Plateau
model:
  name: posUNet3DSparse
optimizer:
  learning_rate: 0.1
  name: SGD
seed: 47
tb_dir: /runs
title: best_generator_exp
trainer:
  checkpoint_path: null
  do_train: true
  epochs: 90
  validate_after_iters: 2
```

In addiction we created a factory for Augmentation which allows you to load augmentations from a yaml file.
the following example can help you to make your own file. In our experiments we just used RandomFlip on all axes.

```yaml
data-loader:
  augmentations_file: /homes/mcipriano/sbatch_scripts/augmentations_files/3.yaml
  background_suppression: 0
  batch_size: 2
  file_path: /nas/softechict-nas-2/mcipriano/datasets/maxillo/VOLUMES
  labels:
    BACKGROUND: 0
    INSIDE: 1
  mean: 0.08435
  num_workers: 4
  patch_shape:
  - 120
  - 120
  - 120
  resize_shape:
  - 168
  - 280
  - 360
  sparse_path: /nas/softechict-nas-2/mcipriano/datasets/maxillo/SPARSE
  split_filepath: /nas/softechict-nas-2/mcipriano/splits/generator_training.json
  split_volumes:
  - 1
  - 320
  - 2
  std: 0.17885
  volumes_max: 2100
  volumes_min: 0
  weights:
  - 0.000703
  - 0.999
loss:
  name: Jaccard
lr_scheduler:
  name: Plateau
model:
  name: posUNet3DSparse
optimizer:
  learning_rate: 0.1
  name: SGD
seed: 47
tb_dir: /nas/softechict-nas-2/mcipriano/runs
title: 6GK1I26CS_CROP
trainer:
  checkpoint_path: null
  do_train: true
  epochs: 90
  validate_after_iters: 2
```

## Directories
Each experiment is expected to be placed into a result dir:

```
results/
├─ experiment_name/
│  ├─ checkpoints/
│  ├─ logs/
│  │  ├─ config.yaml
│  ├─ numpy/

```
If experiment_name does not exist, python will look for a *config.yaml* file in a *config* folder in your project directory.
