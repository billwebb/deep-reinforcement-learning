[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Project 3: Collaboration and Competition

**Project Submission by Bill Webb**

**April 2020**

**Project 3** - Using the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.  This was done on an Windows target using Anaconda Powershell.

1. Create and activate a new environment with Python 3.6.

``` bash
conda create --name pytorch python=3.6
conda activate pytorch
```

2. Clone the repository, and navigate to the python/ folder. This includes the Unity environment for Windows.

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
python -u .\p3.py 2>%1 > .\p3.log
```

Output will go to the p3.log file.
