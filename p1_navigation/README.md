[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

**Project Submission by Bill Webb**

**March 2019**

**Project 1** - Using the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13.  The problem is solved with a average score of +13 after approximately 500 episodes.

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.  This was done on an Ubuntu target.

1. Create and activate a new enivronment with Python 3.6.

``` bash
conda create --name drlnd python=3.6
source activate drlnd
```

2. Clone the repository, and navigate to the python/ folder. Then, install several dependencies, including Pytorch.

``` bash
git clone https://github.com/billwebb/deep-reinforcement-learning
cd deep-reinforcement-learning/python
pip install .
```

3. Install Unity following these instructions - https://github.com/billwebb/deep-reinforcement-learning

4. Create an IPython kernel for the drlnd environment.

``` bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.


### Instructions

Follow the instructions in `Navigation.ipynb` for running the solution to this project.  

