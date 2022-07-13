import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import copy
import time

from dqn.memory import ReplayMemory
from dqn.net import MLP
from dqn.agent import Agent
from dqn.learner import Learner

import optax
from jax import random as jrandom, numpy as jnp
import haiku as hk

seed = 666
key = jrandom.PRNGKey(seed)

env = gym.make('CartPole-v1')
plt.ion()

obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The observation space size: {}".format(obs_space.shape))
print("The action space: {}".format(action_space))
print("The action space size: {}".format(action_space.n))

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))

BATCH_SIZE = 128
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = env.action_space.n
n_obs = obs_space.shape[0]

units = [n_obs * 2, n_actions]

net = MLP(units)
net_transform = hk.without_apply_rng(hk.transform(net))
optimizer = optax.chain(optax.clip(1), optax.rmsprop(learning_rate=1e-2))


print(hk.experimental.tabulate(net_transform)(jnp.zeros((1, n_obs), dtype=np.float32)))

agent = Agent(net_transform.apply)
learner = Learner(agent, optimizer)

key_init, key = jrandom.split(key, 2)
params = net_transform.init(key_init, jnp.zeros((1, n_obs), dtype=np.float32))

params_policy = copy.deepcopy(params)
params_target = copy.deepcopy(params)

opt_state = optimizer.init(params_policy)

memory = ReplayMemory(10000)

episode_durations = []
losses = []

def moving_average(a, m=3):
    n = len(a)
    ret = np.cumsum(a, dtype=float)
    ret[m:] = ret[m:] - ret[:n-m]
    return ret[m - 1:] / m

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = np.array(episode_durations)
    plt.title('Jax Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = moving_average(durations_t, 100)
        means = np.concatenate((np.zeros(99, dtype=np.float32), means))
        plt.plot(means)
    plt.pause(0.001)

def plot_losses():
    plt.figure(3)
    plt.clf()
    losses_array = np.array(losses)
    plt.title('Jax Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(losses_array)
    # Take 100 episode averages and plot them too
    if len(losses_array) >= 100:
        means = moving_average(losses_array, 100)
        means = np.concatenate((np.zeros(99, dtype=np.float32), means))
        plt.plot(means)
    plt.pause(0.001)

num_episodes = 250

def batched_trajs(trajs):
    batched_trajs = Transition(*zip(*trajs))
    states = jnp.asarray(np.stack(batched_trajs.state, axis=0))
    actions = jnp.asarray(np.stack(batched_trajs.action, axis=0))
    rewards = jnp.asarray(np.stack(batched_trajs.reward, axis=0))
    dones = jnp.asarray(np.stack(batched_trajs.done, axis=0))
    next_states = jnp.asarray(np.stack(batched_trajs.next_state, axis=0))
    return states, actions, rewards, dones, next_states

start = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = jnp.asarray(state)

    for t in count():
        # Select and perform an action

        key_explore, key = jrandom.split(key, 2)
        action = agent.explore(params_policy, key_explore, state)
        obs, reward, done, _ = env.step(np.asarray(action))

        # Observe new state
        if not done:
          next_state = np.array(obs)
        else:
          next_state = np.zeros(shape=obs_space.shape, dtype=np.float32)

        done = np.array(done)

        # Store the transition in memory
        memory.push(state, action, done, next_state, reward)

        # Perform one step of the optimization (on the policy network)
        if len(memory) > BATCH_SIZE:

          trajs = memory.sample(BATCH_SIZE)
          states, actions, rewards, dones, next_states = batched_trajs(trajs)

          params_policy, opt_state, loss = learner.update(params_policy, params_target, opt_state,
                                                                   states, actions, rewards, dones, next_states)

          losses.append(loss)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            plot_losses()
            break
        else:
          state = jnp.asarray(next_state)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
      params_target = copy.deepcopy(params_policy)

print('Complete')
print('Time: {}'.format(time.time() - start))
plt.ioff()
plt.show()