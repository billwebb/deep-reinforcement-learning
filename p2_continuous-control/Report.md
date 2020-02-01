## Project 2: Continuous control

**Project Submission by Bill Webb**

**February 2020**

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

### Learning Algorithm

Sources
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
- Udacity DDPG agent implementations and examples

The Deep Deterministic Policy Gradient (DDPG) algorithm is utilized to solve this project.  DDPG is a model-free, off-policy actor-critic algorithm that utilizes deep function approximators.  The advantages in applying this algorithm to the Continuous Control problem include:

- Ideal for continuous and high dimensional action and state space
- Actor-Critic
-- Actor - policy-based (better for continuous) - determines agents action
-- Critic - value-based - provides feedback about the action
-- Increases stability and reduces bias of the algorithm
- Model-Free - to support high dimensionality, estimates the optimal policy and value functions rather than determining the dynamics of the environment
- Off-Policy - slowly updates active policy based on learned policy to maintain stability of the agent

Several improvements were implemented to ensure the best performance of the algorithm, as outlined in the following sections.

#### Deep Learning

There are two neural network models, one each for the Actor and Critic.  In addition, there is a local and target instance of each model, in order to support off-policy learning.

The architecture follows that from the DDPG paper, Section 7 (Lillicrap, et al., 2016).

##### Actor Model

The Actor is a policy-based model which is used by the agent to determine actions.  The following model is defined for the project:

- Input - 33 states, from the environment
- Layer 1 - Linear - 400 nodes
- Batch Normalization Layer
- Layer 2 - Linear - 300 nodes
- Output - Linear - 4 action size
- Tanh activation - to bound the actions

The NN returns the torque to apply to the two joints of the Reacher environment.

##### Critic Model

The Actor is a value-based model which is used by the agent to improve.  The following model is defined for the project:

- Input - 33 states, from the environment
- Layer 1 - Linear - 400 nodes
- Batch Normalization Layer
- Layer 2 - Linear - 300 nodes - Actions included
- Output - Linear - 1 action size
- ReLU activation - due to sparsity

The NN returns whether the action taken was good or bad as a feedback loop.

##### Hyperparameters

The hyperparameters presented in the DDPG paper for low-dimensional given the 33 state size, and provided the required results.

- Buffer size - int(1e6)
- Batch size - 64
- Discount factor (Gamma) - 0.99
- Soft update for target (off-policy) - 1e-3
- Actor learning rate - 1e-4
- Critic learning rate - 1e-3
- Learning timestep interval - 20
- Learning number - 10
- Ornstein-Uhlenbeck noise Sigma - 0.2
- Ornstein-Uhlenbeck noise Theta - 0.15
- Noise step Epsilon - 1.0
- Noise process decay rate - 1e-6

##### Soft Updates

Two copies - regular & target
Target updates with soft update strategy - slowing blending regular with target; every timestep only mix in 0.01% of regular to target.  Faster convergence.
DDPG uses an off-policy method, where both the Actor and Critic each have a regular/local and target network.  In order to increase stability during the learning process, the target network is used to select actions when learning with the replay buffer samples.  Then, the updates based on loss are done to the regular network rather than the regular network.  The weights of the regular network are slowly blended into the target network with a factor of 0.001 per learning step.  This allows the target network to slowly, but regularly, converge to the optimal weights but retain model stability.

This contrasts with other approaches, where the regular network is copied to the target network every X timesteps.  This approach typically leads to model instability and/or increased learning time.

##### Replay Buffer

Experience replay is utilized to prevent correlation between sequence of steps.  In additional, it allows learning from rare step tuples multiple times.  As each step is taken, the step tuple is added to the replay buffer.  Then, during the learning step, a batch of random tuples are chosen to build the target model then backpropogate the error in the current model.

##### Exploration and Noise

The agent has not prior knowledge of the transition model.  In order to find the optimal model, the agent must continue to level of exploration to discovery better, optimal actions/transitions.  During the selection of an action by the agent, the Ornstein-Uhlenbeck noise process is applied to the deterministic continuous action result.  This will result in actions taken by the agent that vary enough from the model explore and possibly discovery better actions.

#### Ideas for Future Work

- Distributed training with 20 agents was utilized in order to increase training efficiency.  Additional agents could be utilized to further incrased efficiency.

- A low-dimensional network was utilized, as the environment included 33 states, which includes the position and velocity of the joints, etc.  When taking the model to a real-world/physical agent, this state information may not be available.  Therefore, it would be desirable to apply DDPG based off of camera pixels, resulting in a high-dimensional network.

### Results

Full results can be found in the `nohup.out` file.  The problem was solved in 103 episodes, with an average reward of at least +30 over the next 100 episodes.

![Results][results.png]

```
Episode 189 (213 sec)  --     Min: 35.6       Max: 39.0       Mean: 37.1      Mov. Avg: 27.9
Episode 190 (213 sec)  --     Min: 27.9       Max: 39.3       Mean: 36.8      Mov. Avg: 28.1
Episode 191 (213 sec)  --     Min: 30.1       Max: 39.4       Mean: 35.9      Mov. Avg: 28.3
Episode 192 (213 sec)  --     Min: 27.4       Max: 39.5       Mean: 35.8      Mov. Avg: 28.5
Episode 193 (213 sec)  --     Min: 27.9       Max: 39.2       Mean: 35.8      Mov. Avg: 28.7
Episode 194 (213 sec)  --     Min: 26.1       Max: 37.9       Mean: 33.4      Mov. Avg: 28.8
Episode 195 (213 sec)  --     Min: 25.8       Max: 39.3       Mean: 33.6      Mov. Avg: 28.9
Episode 196 (213 sec)  --     Min: 31.2       Max: 39.4       Mean: 37.2      Mov. Avg: 29.1
Episode 197 (213 sec)  --     Min: 30.1       Max: 38.6       Mean: 35.1      Mov. Avg: 29.2
Episode 198 (213 sec)  --     Min: 26.2       Max: 38.6       Mean: 35.8      Mov. Avg: 29.4
Episode 199 (213 sec)  --     Min: 28.8       Max: 36.8       Mean: 34.3      Mov. Avg: 29.5
Episode 200 (213 sec)  --     Min: 28.5       Max: 35.5       Mean: 32.8      Mov. Avg: 29.7
Episode 201 (213 sec)  --     Min: 26.8       Max: 38.7       Mean: 35.6      Mov. Avg: 29.8
Episode 202 (213 sec)  --     Min: 32.2       Max: 38.5       Mean: 36.4      Mov. Avg: 29.9
Episode 203 (213 sec)  --     Min: 27.0       Max: 37.7       Mean: 34.0      Mov. Avg: 30.0

Environment SOLVED in 103 episodes!     Moving Average =30.0 over last 100 episodes
```
