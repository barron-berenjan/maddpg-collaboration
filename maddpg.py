import numpy as np
import config as cfg
from ddpg_agent import Agent
from OUNoise import OUNoise
import torch
import torch.nn.functional as F
from replaybuffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class maddpg():

    def __init__(self, state_size, action_size, random_seed, n_agents):

        self.CONFIG = cfg.CONFIG

        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.n_agents = n_agents

        # counter
        self.t_step = 0

        # update counter
        self.u_step = 0

        # replay buffer
        self.memory = ReplayBuffer(action_size, self.CONFIG['BUFFER_SIZE'], self.CONFIG['BATCH_SIZE'], self.random_seed)

        # OUNoise Noise
        self.noise = OUNoise(action_size, random_seed, sigma=self.CONFIG['SIGMA'])

        # ddpg agents
        self.maddpg_agent = [Agent(self.state_size, self.action_size, self.random_seed, self.n_agents) for i in
                             range(self.n_agents)]

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def get_local_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        local_actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return local_actors

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""

        states = torch.from_numpy(states).float().to(device)

        # if self.t_step < self.CONFIG['WARMUP']:
        #     return np.clip([np.random.normal(scale=0.1, size=(self.action_size))
        #                     for n in range(self.n_agents)], -1, 1)

        actions = []
        for idx, agent in enumerate(self.maddpg_agent):
            agent.actor_local.eval()
            with torch.no_grad():
                action = agent.actor_local(states[idx]).cpu().data.numpy()
            agent.actor_local.train()
            actions.append(action)
        actions = np.vstack(actions)
        if add_noise:
            # action += np.random.normal(0, 0.01, size=action.shape)
            actions += self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1

        if (len(self.memory) > self.CONFIG['BATCH_SIZE']):
            for a_i in range(self.n_agents):
                experiences = self.memory.sample()
                self.learn(experiences, self.CONFIG['GAMMA'], a_i)
                self.update_targets()

    def learn(self, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # get action prediction for each target_actor / next_state combination
        actions_next = torch.zeros((self.CONFIG['BATCH_SIZE'], self.n_agents, self.action_size)).to(device)
        policy_target = self.get_target_actors()
        for idx, agent_t in enumerate(policy_target):
            actions_next[:, idx, :] = agent_t(next_states[:, idx, :])
        actions_next = torch.reshape(actions_next, (actions_next.shape[0], -1))  # reshape to flatten actions

        ## get q_values for the target critic for these actions...

        next_states_ = torch.reshape(next_states, (next_states.shape[0], -1))
        Q_targets_next = agent.critic_target(next_states_, actions_next)

        #### CRITIC LOSS ###

        Q_targets = rewards[:, agent_number].view(-1, 1) + (
                gamma * Q_targets_next * (1 - dones[:, agent_number].view(-1, 1)))

        # calculate Q_expected
        states_ = torch.reshape(states, (states.shape[0], -1))
        actions_ = torch.reshape(actions, (actions.shape[0], -1))
        Q_expected = agent.critic_local(states_, actions_)

        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), self.CONFIG["CLAMP_VALUE"])
        agent.critic_optimizer.step()

        #### ACTOR LOSS ###
        # update actor network using policy gradient

        agent.actor_optimizer.zero_grad()

        # current_policy_out = agent.actor_local(states[:,agent_number,:])

        q_input = [self.maddpg_agent[i].actor_local(ob) if i == agent_number \
                       else self.maddpg_agent[i].actor_local(ob).detach()
                   for i, ob in enumerate(states.permute(1, 0, 2))]

        q_input = torch.cat(q_input, dim=1)

        #### ANOTHER WAY OF DOING THE SAME ABOVE ####

        # q_input = actions.clone()
        # agents_state = states[:,agent_number,:]
        # q_input[:, agent_number, :] = agent.actor_local(agents_state)
        # q_input = q_input.permute(1, 0, 2)
        # q_input_ = torch.cat([q_input[i] for i in range(q_input.shape[0])], dim=1)

        actor_loss = -agent.critic_local(states_, q_input).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), self.CONFIG["CLAMP_VALUE"])
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""

        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.CONFIG['TAU'])
            self.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.CONFIG['TAU'])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()