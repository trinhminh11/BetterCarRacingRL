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
from tqdm import tqdm

import better_car_racing
import gym_agent as ga

# Local imports
import utils


import PPO


class Linear_Actor(nn.Module):
    def __init__(self, n_inp, n_actions, features = [256, 256]):
        super().__init__()

        layer_sizes = [n_inp] + features + [n_actions]

        # Initialize an empty sequential container
        self.net = nn.Sequential()

        # Loop through the layer sizes to create the network
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer
            self.net.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add ReLU activation function for all layers except the last one
            if i != len(layer_sizes) - 2:
                self.net.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class Linear_Critic(nn.Module):
    def __init__(self, n_inp, features = [256, 256]):
        super().__init__()

        layer_sizes = [n_inp] + features + [1]

        # Initialize an empty sequential container
        self.net = nn.Sequential()

        # Loop through the layer sizes to create the network
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer
            self.net.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add ReLU activation function for all layers except the last one
            if i != len(layer_sizes) - 2:
                self.net.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)
    
class Linear_Policy(nn.Module):
    def __init__(self, n_inp, n_actions: int, features: list[int] = [256, 256], optimizer = optim.Adam, actor_lr = 5e-4, critic_lr=5e-4, optimizer_kwargs = None):
        super().__init__()

        self.actor = Linear_Actor(n_inp, n_actions, features)
        self.critic = Linear_Critic(n_inp, features)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.actor_optimizer = optimizer(self.actor.parameters(), lr=actor_lr, **optimizer_kwargs)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=critic_lr, **optimizer_kwargs)

class CNN_Actor(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, filters: list[int] = [16, 32, 64], fc: list[int] = [256]) -> None:
        super().__init__()

        self.initial = utils.ConvBn(in_channels, filters[0], 8, 4, 2)

        self.conv = nn.Sequential()

        for i in range(len(filters)-1):
            self.conv.append(utils.ConvBn(filters[i], filters[i+1], pool=True))
        
        self.conv.append(nn.AdaptiveMaxPool2d((4, 4)))

        self.conv.append(nn.Flatten())

        self.fcs = nn.Sequential(
            nn.Linear(4*4*filters[-1], fc[0]),
            nn.ReLU(),
            nn.Linear(fc[0], n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        image_encoded = self.initial(state)
        image_encoded = self.conv(image_encoded)

        return self.fcs(image_encoded)

class CNN_Critic(nn.Module):
    def __init__(self, in_channels: int, filters: list[int] = [16, 32, 64], fc: list[int] = [256]) -> None:
        super().__init__()

        self.initial = utils.ConvBn(in_channels, filters[0], 8, 4, 2)

        self.conv = nn.Sequential()

        for i in range(len(filters)-1):
            self.conv.append(utils.ConvBn(filters[i], filters[i+1], pool=True))
        
        self.conv.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.conv.append(nn.Flatten())

        self.fcs = nn.Sequential(
            nn.Linear(filters[-1], fc[0]),
            nn.ReLU(),
            nn.Linear(fc[0], 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        image_encoded = self.initial(state)
        image_encoded = self.conv(image_encoded)

        return self.fcs(image_encoded)

class CNN_Policy(nn.Module):
    def __init__(self, in_channels, n_actions, filters, fcs, optimizer = optim.Adam, actor_lr = 5e-5, critic_lr=1e-4, optimizer_kwargs = None):
        super().__init__()

        self.actor = CNN_Actor(in_channels, n_actions, filters, fcs)
        self.critic = CNN_Critic(in_channels, filters, fcs)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.actor_optimizer = optimizer(self.actor.parameters(), lr=actor_lr, **optimizer_kwargs)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=critic_lr, **optimizer_kwargs)

class Combined_Actor(nn.Module):
    def __init__(self, in_channels: int, n_vector: int, n_actions: int, filters: list[int] = [16, 32, 64], fc: list[int] = [256]) -> None:
        super().__init__()

        self.initial = utils.ConvBn(in_channels, filters[0], pool=False)

        self.conv = nn.Sequential()

        for i in range(len(filters)-1):
            self.conv.append(utils.ConvBn(filters[i], filters[i+1], pool=True))
        
        self.conv.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.conv.append(nn.Flatten())

        self.fcs = nn.Sequential(
            nn.Linear(filters[-1] + n_vector, fc[0]),
            nn.ReLU(),
            nn.Linear(fc[0], n_actions)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        image = state['image']
        vector = state['vector']

        image_encoded = self.initial(image)
        image_encoded = self.conv(image_encoded)

        inp_vector = torch.cat([image_encoded, vector], dim=1)

        return self.fcs(inp_vector)

class Combined_Critic(nn.Module):
    def __init__(self, in_channels: int, n_vector: int, filters: list[int] = [16, 32, 64], fc: list[int] = [256]) -> None:
        super().__init__()

        self.initial = utils.ConvBn(in_channels, filters[0], pool=False)

        self.conv = nn.Sequential()

        for i in range(len(filters)-1):
            self.conv.append(utils.ConvBn(filters[i], filters[i+1], pool=True))
        
        self.conv.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.conv.append(nn.Flatten())

        self.fcs = nn.Sequential(
            nn.Linear(filters[-1] + n_vector, fc[0]),
            nn.ReLU(),
            nn.Linear(fc[0], 1)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        image = state['image']
        vector = state['vector']

        image_encoded = self.initial(image)
        image_encoded = self.conv(image_encoded)

        inp_vector = torch.cat([image_encoded, vector], dim=1)

        return self.fcs(inp_vector)

class Combined_Policy(nn.Module):
    def __init__(self, in_channels, n_vector: int, n_actions, filters, fcs, optimizer = optim.Adam, actor_lr = 5e-5, critic_lr=1e-4, optimizer_kwargs = None):
        super().__init__()

        self.actor = Combined_Actor(in_channels, n_vector, n_actions, filters, fcs)
        self.critic = Combined_Critic(in_channels, n_vector, filters, fcs)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.actor_optimizer = optimizer(self.actor.parameters(), lr=actor_lr, **optimizer_kwargs)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=critic_lr, **optimizer_kwargs)

# Transform
class StateTfm(ga.Transform):
    def __init__(self, net_type = 'linear', n_frames = 4):
        super().__init__()

        self.net_type = net_type

        self.n_frame = n_frames

        if net_type == 'linear':

            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(14, ), 
                dtype=np.float32
            )
        else:
            self.frames = {'image': np.zeros((self.n_frame, 96, 96), dtype=np.float32), 'vector': np.zeros((7, ), dtype=np.float32)}

            self.start = False
            if net_type == 'cnn':
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_frames, 96, 96), dtype=np.float32)
            elif net_type == 'combine':
                self.observation_space = gym.spaces.Dict({
                    'image': gym.spaces.Box(low=0, high=1, shape=(n_frames, 96, 96), dtype=np.float32),
                    'vector': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7, ), dtype=np.float32),
                })
            else:
                raise ValueError('Invalid net_type')

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.net_type == 'linear':
            rays = observation['rays'].astype(np.float32)
            vels = observation['vels'].astype(np.float32)
            # pos_flag = observation['pos_flag'].astype(np.float32)

            res = np.concatenate([rays, vels], axis=-1)
            return res
        else:
            image = observation['image'].astype(np.float32).transpose([2, 0, 1]) # n_envs, 3, 96, 96
            r, g, b = image[0], image[1], image[1]
            gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

            vels = observation['vels'].astype(np.float32)
            # res = np.concatenate([rays, vels], axis=-1)

            if self.start:
                self.frames['image'][0] = gray_image
                self.frames['image'] = np.roll(self.frames['image'], shift=-1, axis=0)
                
            else:
                for i in range(self.n_frame):
                    self.frames['image'][i] = gray_image

            self.frames['vector'] = vels

            self.start = True

            if self.net_type == 'cnn':
                return self.frames['image']
            elif self.net_type == 'combine':
                return self.frames
            else:
                raise ValueError('Invalid net_type')

    def reset(self, **kwargs):
        if self.net_type == 'cnn' or self.net_type == 'combine':
            self.frames = {'image': np.zeros((self.n_frame, 96, 96), dtype=np.float32), 'vector': np.zeros((7, ), dtype=np.float32)}

            self.start = False
    
class ActionTfm(ga.Transform):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(4)

    def __call__(self, action: int) -> int:
        return action + 1
        # return action

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            if self.unwrapped.render_mode == 'human':
                self.unwrapped.render()
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


def play(net_type: str, seed: int):
    n_envs = 1
    chkpt_dir = 'checkpoints/BetterCarRacing-v0/PPO'

    env_id = "BetterCarRacing-v0"

    die_if_grass = True

    if net_type == 'linear':
        env = ga.make_vec(env_id, num_envs=n_envs, continuous=False, die_if_grass = die_if_grass, random_direction=False, observation_transform=StateTfm(), action_transform=ActionTfm())
        agent = PPO.PPO(
            name = "PPO",
            policy = Linear_Policy(14, 4, [256, 256], actor_lr = 5e-5, critic_lr=1e-4).apply(utils.init_weights('kaiming')),
            env = env,
            k = 10,
            coef_crit = 1,
            coef_entropy = 0.1,
            batch_size = None,
            gamma = 0.99, 
            lamda = 0.95,
            eps_clip = 0.2,
            clip_vf = 10,
            max_norm = 5,
            force_forward = True,
            device='auto',
            seed=seed
        )

        env = ga.make(env_id, render_mode = 'human', continuous=False, die_if_grass = die_if_grass, random_direction=False, lap_complete_percent = 1, render_ray=False)
        env.set_observation_transform(StateTfm())
        env.set_action_transform(ActionTfm())

    elif net_type == 'cnn':
        env = ga.make(env_id, observation_transform=StateTfm('cnn', n_frames=4), continuous=False, die_if_grass = die_if_grass, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        agent = PPO.PPO(
            name = 'CNN_PPO',
            policy = CNN_Policy(4, 5, filters=[16, 32, 64], fcs=[256]).apply(utils.init_weights('kaiming')),
            env = env,
            k = 10,
            coef_crit = 1,
            coef_entropy = 0.1,
            batch_size = None,
            gamma = 0.99, 
            lamda = 0.95,
            eps_clip = 0.2,
            clip_vf = 10,
            max_norm = 5,
            force_forward = True,
            device='auto',
            seed=seed
        ).to('cuda')

        env = ga.make(env_id, render_mode = 'human', observation_transform=StateTfm('cnn', n_frames=4), continuous=False, die_if_grass = True, lap_complete_percent = 1, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
    
    elif net_type == 'combine':
        env = ga.make(env_id, observation_transform=StateTfm('combine'), continuous=False, die_if_grass = die_if_grass, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        agent = PPO.PPO(
            name = 'COMBINED_PPO',
            policy = Combined_Policy(4, 7, 5, filters=[16, 32, 64], fcs=[256]).apply(utils.init_weights('kaiming')),
            env = env,
            k = 10,
            coef_crit = 1,
            coef_entropy = 0.1,
            batch_size = None,
            gamma = 0.99, 
            lamda = 0.95,
            eps_clip = 0.2,
            clip_vf = 10,
            max_norm = 5,
            force_forward = True,
            device='auto',
            seed=seed
        ).to('cuda')

        env = ga.make(env_id, render_mode = 'human', observation_transform=StateTfm('combine', n_frames=4), continuous=False, die_if_grass = True, lap_complete_percent = 1, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        
    
    agent.load(chkpt_dir, 'best')

    agent.play(env, stop_if_truncated=True, seed = seed)


def main():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help'
    )
    
    parser.add_argument('--net_type', type=str, default='linear', help='Type of network to use')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')

    args = parser.parse_args()

    print(args.net_type)
    print(args.seed)

    play(args.net_type.lower(), args.seed)

if __name__ == "__main__":
    main()
