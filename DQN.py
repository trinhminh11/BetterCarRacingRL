# Standard library imports
import random

# Third-party imports
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import gym_agent as ga


class DQN(ga.OffPolicyAgent):
    # policy: Policy
    def __init__(
            self, 
            policy, 
            env,    
            action_space,
            gamma = 0.99,
            eps_start = 1.0,
            eps_decay = 0.995,
            eps_end = 0.01,
            tau = 1e-3,
            batch_size = 64, 
            update_every = 1, 
            name = None,
            device='auto', 
            seed=None
        ):

        super().__init__(policy, env, gamma=gamma, batch_size=batch_size, update_every=update_every, name = name, device=device, seed=seed)

        self.action_space = action_space

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        # Soft update parameter
        self.tau = tau

        self.target_policy = policy

        # Initialize epsilon for epsilon-greedy policy
        self.eps = eps_start

        self.soft_update(1)

    def reset(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    @torch.no_grad()
    def predict(self, state: np.ndarray | dict[str, np.ndarray], deterministic = True) -> int:
        # Determine epsilon value based on evaluation mode
        if deterministic:
            eps = 0
        else:
            eps = self.eps
        
        rng = random.random()

        if isinstance(state, dict):
            if rng >= eps:
                _state = {}

                for key, value in state.items():
                    _state[key] = torch.from_numpy(value).float().to(self.device)

                # Set local model to evaluation mode
                self.policy.eval()
                # Get action values from the local model
                action_value = self.policy.forward(_state)
                # Set local model back to training mode
                self.policy.train()

                # Return the action with the highest value
                return np.argmax(action_value.cpu().data.numpy(), axis=1)
            else:
                # Return a random action from the action space
                return [random.choice(self.action_space) for _ in range(state['image'].shape[0])]
        else:
            # Epsilon-greedy action selection
            if rng >= eps:
                # Convert state to tensor and move to the appropriate device
                state = torch.from_numpy(state).float().to(self.device)

                # Set local model to evaluation mode
                self.policy.eval()
                # Get action values from the local model
                action_value = self.policy.forward(state)
                # Set local model back to training mode
                self.policy.train()

                # Return the action with the highest value
                return np.argmax(action_value.cpu().data.numpy(), axis=1)
            else:
                # Return a random action from the action space
                ret_action = [self.dummy_env.action_space.sample() for _ in range(state.shape[0])]
                return ret_action
                # return [random.choice(self.action_space) for _ in range(state.shape[0])]

    def learn(self, states: Tensor, actions: Tensor, rewards: Tensor, next_states: Tensor, terminals: Tensor):
        """
        Update the value network using a batch of experience tuples.

        Params
        ======
            states (Tensor): Batch of current states
            actions (Tensor): Batch of actions taken
            rewards (Tensor): Batch of rewards received
            next_states (Tensor): Batch of next states
            terminals (Tensor): Batch of terminal flags indicating episode end
        """

        # Get the maximum predicted Q values for the next states from the target model
        q_targets_next = self.policy.target_forward(next_states).detach().max(1)[0]
        # Compute the Q targets for the current states
        q_targets = rewards + (self.gamma * q_targets_next * (~terminals))

        # Get the expected Q values from the local model
        q_expected = self.policy.forward(states)

        q_expected = q_expected.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        # Compute the loss
        loss = F.smooth_l1_loss(q_expected, q_targets)

        # Minimize the loss
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        # Update the target network
        self.soft_update()

    def soft_update(self, tau = None):
        if tau is None:
            tau = self.tau

        self.policy.soft_update(tau)