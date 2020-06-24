import numpy as np
import random
from model import Actor, Critic
from config import CONFIG
import torch
import torch.optim as optim

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

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size * self.number_of_agents, self.action_size * self.number_of_agents,
                                   random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CONFIG['LR_CRITIC'],
                                           weight_decay=CONFIG['WEIGHT_DECAY'])
        self.critic_target = Critic(self.state_size * self.number_of_agents, self.action_size * self.number_of_agents,
                                    random_seed).to(device)
        