[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 1: Continuous Control

**Project Submission by Bill Webb**

**January 2020**

**Project 2** - Using the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

This project utilized 20 identical agents, each with its own copy of the environment.

### Solving the Environment

The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.  This was done on an Ubuntu target.

1. Create and activate a new enivronment with Python 3.6.

``` bash
conda create --name pytorch python=3.6
source activate pytorch
```

2. Clone the repository, and navigate to the python/ folder. This includes the Unity environment for Linux that includes 20 agents.

Then, install several dependencies.

``` bash
git clone https://github.com/billwebb/deep-reinforcement-learning
cd deep-reinforcement-learning/python
pip install .
```
3. Install Pytorch with CUDA support, since GPU acceleration is being used.

``` bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### Instructions

The training process takes many hours, therefore the project has been implemented as a Python script.  This allows it to run in the background on the GPU-based server uninterrupted.

To run the project, run the following -

``` bash
nohup python -u p2.py &
```

Output will go to the nohup.out file.

Then, to monitor the status, run the following -

``` bash
tail -f nohup.out
```
