from unityagents import UnityEnvironment
import numpy as np
from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import datetime

my_env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')
#my_env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

brain_name = my_env.brain_names[0]
brain = my_env.brains[brain_name]
env_info = my_env.reset(train_mode=True)[brain_name]

# number of agents
print('Number of agents:', len(env_info.agents))

# size of each action
print('Size of each action:', brain.vector_action_space_size)
print('Number of states:', env_info.vector_observations.shape[1])

def my_ddpg(my_env, agent, n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    all_scores = []

    print('\r{} START'.format(datetime.datetime.now()))
    for i_episode in range(1, n_episodes+1):
        env_info = my_env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        agent.reset()
        scores = np.zeros(len(env_info.agents))
        for t in range(max_t):
            actions = agent.act(states)
            #next_state, reward, done, _ = my_env.step(action)
            env_info = my_env.step(actions)[brain_name]        # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += rewards                                  # update the score (for each agent)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            if np.any(dones):
                break
        mean_score = np.mean(scores)
        scores_deque.append(mean_score)
        all_scores.append(mean_score)
        print('\r{}\tEpisode {}\tAverage Score: {:.2f}'.format(datetime.datetime.now(), i_episode, mean_score), end="")
        #torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        #torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\r{}\tEpisode {}\tAverage Score: {:.2f}'.format(datetime.datetime.now(), i_episode, mean_score))

    return all_scores

print("calling ddpg")
scores = my_ddpg(my_env, Agent(state_size=env_info.vector_observations.shape[1],
                               action_size=brain.vector_action_space_size, random_seed=2))
