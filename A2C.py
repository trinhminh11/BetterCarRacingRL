# Standard library imports
import random
from typing import Type

# Third-party imports
import torch
from torch import Tensor, nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm.autonotebook import tqdm

import better_car_racing
import gym_agent as ga

# Local imports
import utils

class A2C(ga.OnPolicyAgent):
    def __init__(
            self,
            policy,
            env,
            gamma: float = 0.99,
            gae_lambda: float = 1,
            buffer_size = int(1e5),
            batch_size = None,
            entropy_coef = 0.1,
            name = None,
            device='auto',
            seed=None
        ):
        super().__init__(policy, env, gamma, buffer_size=buffer_size, gae_lambda=gae_lambda, batch_size=batch_size, name=name, device=device, seed=seed)
        self.entropy_coef = entropy_coef
        self.add_optimizer('actor', policy.actor_optimizer)
        self.add_optimizer('critic', policy.critic_optimizer)

    @torch.no_grad()
    def predict(self, states: np.ndarray | dict[str, np.ndarray], deterministic = True):
        if isinstance(states, dict):
            _state = {}

            for key, value in states.items():
                _state[key] = torch.from_numpy(value).float().to(self.device)
        else:
            _state = torch.from_numpy(states).float().to(self.device)

        logits = self.policy.actor(_state)

        probs = F.softmax(logits, dim=1)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()


        return actions.detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(self, states: np.ndarray | dict[str, np.ndarray], deterministic = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(states, dict):
            _state = {}

            for key, value in states.items():
                _state[key] = torch.from_numpy(value).float().to(self.device)
        else:
            _state = states

        logits: Tensor = self.policy.actor(_state)
        values: Tensor = self.policy.critic(_state)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs: Tensor = dist.log_prob(actions)

        return actions.detach().cpu().numpy(), values.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def evaluate_actions(self, states: Tensor | dict[str, Tensor], action: Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy.actor(states)
        values = self.policy.critic(states).squeeze(1)

        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return values, log_probs, entropy
    
    def learn(self, memory: ga.RolloutBuffer):
        for rollout in memory.get(None):
            values, log_prob, entropy = self.evaluate_actions(rollout.observations, rollout.actions)

            advantages = rollout.advantages

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            returns = rollout.returns

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()

            value_loss = F.smooth_l1_loss(values, returns, reduction='mean')

            loss = policy_loss + value_loss - self.entropy_coef * entropy.mean()

            # Optimization step
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            loss.backward()
            self.policy.actor_optimizer.step()
            self.policy.critic_optimizer.step()