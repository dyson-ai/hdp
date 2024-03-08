# Hierarchical Diffusion Policy

This repo contains the PyTorch implementation of the CVPR 2024 paper

Hierarchical Diffusion Policy for Multi-Task Robotic Manipulation

[Xiao Ma](https://yusufma03.github.io/), [Sumit Patidar](https://rocketsumit.github.io/), [Iain Haughton](https://www.linkedin.com/in/iain-haughton-194321135/?originalSubdomain=uk), [Stephen James](https://stepjam.github.io/)

CVPR 2024

Dyson Robot Learning Lab

![teaser](images/sim.gif)

HDP factorises a manipulation policy into a hierarchical structure: a high-level task-planning agent which predicts a distant next-best end-effector pose (NBP), and a low-level goal-conditioned diffusion policy which generates optimal motion trajectories. The factorised policy representation allows HDP to tackle both long-horizon task planning while generating fine-grained low-level actions. To generate context-aware motion trajectories while satisfying robot kinematics constraints, we present a novel kinematics-aware goal-conditioned control agent, Robot Kinematics Diffuser (RK-Diffuser). Specifically, RK-Diffuser learns to generate both the end-effector pose and joint position trajectories, and distill the accurate but kinematics-unaware end-effector pose diffuser to the kinematics-aware but less accurate joint position diffuser via differentiable kinematics.

In this repository, we provide the code for training the low-level RK-Diffuser. We use PerAct as our high-level agent and we refer to its [official implementation](https://github.com/peract/peract).

For more details, see our [project page](https://yusufma03.github.io/projects/hdp/).

## Installation

```bash
conda create -n hdp python=3.10
conda activate hdp
bash ./extra_scripts/install_coppeliasim.sh
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Running in headless mode
Please refer to the official guide of [RLBench](https://github.com/stepjam/RLBench?tab=readme-ov-file#running-headless).

## Generate the dataset
First, we need to generate the training dataset.
```bash
python extra_scripts/dataset_generator.py --save_path=<your dataset path> --tasks=<your task> --variations=1 --processes=1 --episodes_per_task=100
```
For example, to generate a dataset for the `reach_target` task,
```bash
python extra_scripts/dataset_generator.py --save_path=/data/${USER}/rlbench --tasks=reach_target --variations=1 --processes=1 --episodes_per_task=100
```
The script will generate both `train` and `eval` datasets at the same time.

## Training the low-level RK-Diffuser
To train the low-level sim agent, simply do
```bash
python3 train_low_level.py env=sim env.data_path=<your dataset path> env.tasks="[<task1>, <task2>, ...]"
```

For example, to train a model for the `reach_target` and the `take_lid_off_saucepan` tasks, run
```bash
python3 train_low_level.py env=sim env.data_path=/data/${USER}/rlbench env.tasks="[reach_target, take_lid_off_saucepan]"
```

You can enable online logging or set wandb run name by adding the following args
```bash
log=True run_name=<your run name>
```

## Citation

```bibtex
@article{ma2024hierarchical,
  author    = {Ma, Xiao and Patidar, Sumit and Haughton, Iain and James, Stephen},
  title     = {Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation},
  journal   = {CVPR},
  year      = {2024},
}
```
