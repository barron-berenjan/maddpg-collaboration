import numpy as np
import random
import copy
from collections import namedtuple, deque
from copy import deepcopy
from OUNoise import OUNoise
from model import Actor, Critic
from config import CONFIG
import torch
import torch.nn.functional as F
import torch.optim as optim

# BUFFER_SIZE = int(1e6)  # replay buffer size
# BATCH_SIZE = 512        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-2              # for soft update of target parameters
# LR_ACTOR = 1e-3         # learning rate of the actor
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0        # L2 weight decay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, number_of_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.number_of_agents = number_of_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=CONFIG['LR_ACTOR'])
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        # self.actor_target = deepcopy(self.actor_local)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size * self.number_of_agents, self.action_size * self.number_of_agents,
                                   random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CONFIG['LR_CRITIC'],
                                           weight_decay=CONFIG['WEIGHT_DECAY'])
        self.critic_target = Critic(self.state_size * self.number_of_agents, self.action_size * self.number_of_agents,
                                    random_seed).to(device)
        # self.critic_target = deepcopy(self.critic_local)

        # Noise process
        # self.noise = OUNoise(action_size, random_seed, sigma=CONFIG['SIGMA'])

    # def act(self, state, noise=0.0):
    #     state = torch.from_numpy(state).float().to(device)
    #     with torch.no_grad():
    #         action = self.actor_local(state).cpu().data.numpy() + (noise * self.noise.sample())
    #         action = np.clip(action, -1,1)
    #     return action
    #
    #
    # def reset(self):
    #     self.noise.reset()