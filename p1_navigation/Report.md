## Project 1: Navigation

**Project Submission by Bill Webb**

**March 2019**

**Project 1** - Using the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Overview

For this project, we trained an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13.  The problem is solved with a average score of +13 after approximately 500 episodes.


### Learning Algorithm

Deep Q-Learning is used to solve the project.  Q-Learning allows the agent to learn a policy without any knowledge of the environement.  Through succesive episodes, the agent gains feedback through rewards based on the action it takes.  

To improve on the Q-Learning algorithm, Experience Replay and Fixed Q-Targets were used.  A neural network was used for the state-action mapping based on rewards.  The following algorithm was used to solve the problem -

- Initialize replay memory with size 100,000
- Initialize local and target action-value NN with random weights
- For 500 episodes
    - Initialize epsiode and get initial state
    - For each step until done
    - Choose action based on epislon greedy algorithm
    - Take action and get reward and next-state
    - Store experience in replay memory
    - Update to next-step
    - Every 4 steps - perform learning
        - Obtain random batch of 64 from replay memory
        - Set target based on batch samples
        - Calculate error from target vs. current and update with backpropogation

#### Deep Learning

A neural network model is utilized to map the state-action pairs to rewards.  The following model is defined for this project -

- Input - 37 states
- Layer 1 - Linear - 128 nodes
- ReLu activiation
- Layer 2 - Linear - 64 nodes
- ReLu activation
- Output - Linear - 4 actions

The NN returns weights for the 4 actions.  The highest value action is chosen unless the epsilon-greedy algorithm randomly chooses an action for the step.

#### Experience Replay 

Experience replay is utilized to prevent correlation between sequence of steps.  In additional, it allows learning from rare step tuples multiple times.  As each step is taken, the step tuple is added to the replay buffer.  Then, during the learning step, a batch of random tuples are chosen to build the target model then backpropogate the error in the current model.

#### Fixed Q-Targets

Fixed Q-Targets are utilized so the the model currently being used to step through the episode isn't being updated immediately.  We're already beginning with a guess, so an ongoing update of a guess with a guess can result in an erratic model that will give bad results.  At each point we doing the learning step to the target model, we then update the current model relative to the update hyperparameter τ = 0.001.  This provides a weight between the current and target models as -

- θ_target = τ*θ_local + (1 - τ)*θ_target

#### Hyperparameters

- Replay buffer size - 100,000
- Batch size - 64 step tuples randonly selected from replay buffer during each learning step
- Gamma - 0.99 - discount factor applied to subsequent steps.  A higher value means little discouting
- Tau - 0.001 - weight applied to current vs. target models during Fixed Q-Target update.  Small value means target model preferred
- Learning rate - 0.0005 - how much model is updated in proporation to calculated error during backpropogation
- Update Every - 4 - how often to perform learning during the episode.  Allows agent to run against a stable model and make updates in enough data

