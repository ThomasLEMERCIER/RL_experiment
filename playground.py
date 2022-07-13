import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import time
from torchinfo import summary


from dqn.network import DQN



env = gym.make('CartPole-v1')
plt.ion()

obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The observation space size: {}".format(obs_space.shape))
print("The action space: {}".format(action_space))
print("The action space size: {}".format(action_space.n))

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = env.action_space.n
n_obs = obs_space.shape[0]

policy_net = DQN(n_obs, n_obs * 2, n_actions, device).to(device)
target_net = DQN(n_obs, n_obs * 2, n_actions, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print(summary(policy_net))

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            Qvalues = policy_net(state)
            action = torch.argmax(Qvalues, dim=-1)
            return action.view(1)
            
    else:
        return torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long)


episode_durations = []
losses = []

def moving_average(a, m=3):
    n = len(a)
    ret = np.cumsum(a, dtype=float)
    ret[m:] = ret[m:] - ret[:n-m]
    return ret[m - 1:] / m

def plot_losses():
    plt.figure(3)
    plt.clf()
    losses_array = np.array(losses)
    plt.title('Pytorch Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(losses_array)
    # Take 100 episode averages and plot them too
    if len(losses_array) >= 100:
        means = moving_average(losses_array, 100)
        means = np.concatenate((np.zeros(99, dtype=np.float32), means))
        plt.plot(means)
    plt.pause(0.001)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = np.array(episode_durations)
    plt.title('Pytorch Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = moving_average(durations_t, 100)
        means = np.concatenate((np.zeros(99, dtype=np.float32), means))
        plt.plot(means)
    plt.pause(0.001)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state, dim=0)
    action_batch = torch.stack(batch.action, dim=0)
    reward_batch = torch.stack(batch.reward, dim=0)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    losses.append(loss.item())

start = time.time()
num_episodes = 250
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state).float()

    for t in count():
        # Select and perform an action
        
        action = select_action(state)
        obs, reward, done, _ = env.step(action.item())
        reward = torch.tensor(reward)

        # Observe new state
        if not done:
            next_state = torch.tensor(obs)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            plot_losses()
            break
        else:
          state = next_state.clone().detach()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
print('Time: ', time.time() - start)
plt.ioff()
plt.show()