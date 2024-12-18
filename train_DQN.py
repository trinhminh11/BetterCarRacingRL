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

import DQN


class Linear_DQN(nn.Module):
    def __init__(self, n_inp: int, features: list[int], n_actions: int):
        super().__init__()

        # Create a list of layer sizes including input, hidden, and output layers
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Pass the input state through the network
        return self.net(state)

class Linear_Policy(nn.Module):
    def __init__(self, n_inp: int, features: list[int], n_actions: int, optimizer: Type[optim.Optimizer] = optim.Adam, lr: float = 5e-4, optimizer_kwargs: dict = None):
        super().__init__()
        self.network = Linear_DQN(n_inp, features, n_actions)
        self.target_network = Linear_DQN(n_inp, features, n_actions)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        self.optimizer = optimizer(self.network.parameters(), lr=lr, **optimizer_kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def target_forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_network(state)

    def soft_update(self, tau: float):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class CNN_DQN(nn.Module):
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

class CNN_Policy(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, filters: list[int], fcs: list[int], optimizer: Type[optim.Optimizer] = optim.Adam, lr: float = 5e-4, optimizer_kwargs: dict = None):
        super().__init__()
        self.network = CNN_DQN(in_channels, n_actions, filters, fcs)
        self.target_network = CNN_DQN(in_channels, n_actions, filters, fcs)
        self.soft_update(tau=1)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        self.optimizer = optimizer(self.network.parameters(), lr=lr, **optimizer_kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def target_forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_network(state)

    def soft_update(self, tau: float):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class Combine_DQN(nn.Module):
    def __init__(self, in_channels: int, n_vector: int, n_actions: int, filters: list[int] = [32, 64, 128], fcs: list[int] = [256]) -> None:
        super().__init__()

        self.initial = utils.ConvBn(in_channels=in_channels, out_channels=filters[0], pool=False)

        self.conv = nn.Sequential()

        for i in range(len(filters)-1):
            self.conv.append(utils.ConvBn(in_channels=filters[i], out_channels=filters[i+1], pool=True))
        
        self.conv.append(nn.AdaptiveMaxPool2d((1, 1)))

        self.conv.append(nn.Flatten())

        self.fcs = nn.Sequential(
            nn.Linear(filters[-1] + n_vector, fcs[0]),
            nn.ReLU(),
            nn.Linear(fcs[0], n_actions)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        image = state['image']
        vector = state['vector']

        image_encoded = self.initial(image)
        image_encoded = self.conv(image_encoded)

        inp_vector = torch.cat([image_encoded, vector], dim=1)

        return self.fcs(inp_vector)
    
class Combine_Policy(nn.Module):
    def __init__(self, in_channels: int, n_vector: int, n_actions: int, filters: list[int], fcs: list[int], optimizer: Type[optim.Optimizer] = optim.Adam, lr: float = 5e-4, optimizer_kwargs: dict = None):
        super().__init__()
        self.network = Combine_DQN(in_channels, n_vector, n_actions, filters, fcs)
        self.target_network = Combine_DQN(in_channels, n_vector, n_actions, filters, fcs)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        self.optimizer = optimizer(self.network.parameters(), lr=lr, **optimizer_kwargs)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(state)

    def target_forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.target_network(state)

    def soft_update(self, tau: float):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



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


def main():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help'
    )
    
    parser.add_argument('--net_type', type=str, default='linear', help='Type of network to use')

    args = parser.parse_args()

    net_type = args.net_type
    print(net_type)
    n_envs = 1
    chkpt_dir = 'checkpoints/BetterCarRacing-v0/DQN'

    env_id = "BetterCarRacing-v0"

    seed = 110404

    die_if_grass = True

    if net_type == 'linear':
        env = ga.make_vec(env_id, num_envs=n_envs, continuous=False, die_if_grass = die_if_grass, random_direction=False, observation_transform=StateTfm(), action_transform=ActionTfm())
        agent = DQN.DQN(
            name = 'DQN',
            policy = Linear_Policy(14, [256, 256], 4).apply(utils.init_weights('kaiming')),
            env = env,
            action_space = list(range(4)),
            gamma=0.99,
            eps_start=1.0,
            eps_decay=0.995,
            eps_end=0.01,
            tau=1e-3,
            batch_size=64,
            update_every=4,
            device='auto',
            seed=seed
        ).to('cuda')

        env = ga.make(env_id, render_mode = 'human', continuous=False, die_if_grass = die_if_grass, random_direction=False, lap_complete_percent = 1, render_ray=True)
        env.set_observation_transform(StateTfm())
        env.set_action_transform(ActionTfm())

    elif net_type == 'cnn':
        env = ga.make(env_id, observation_transform=StateTfm('cnn', n_frames=4), continuous=False, die_if_grass = die_if_grass, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        agent = DQN.DQN(
            name = 'CNN_DQN',
            policy = CNN_Policy(4, 5, filters=[16, 32, 64], fcs=[256]).apply(utils.init_weights('kaiming')),
            env = env,
            action_space = list(range(5)),
            gamma=0.99,
            eps_start=1.0,
            eps_decay=0.995,
            eps_end=0.01,
            tau=1e-3,
            batch_size=64,
            update_every=4,
            device='auto',
            seed=seed
        ).to('cuda')

        env = ga.make(env_id, render_mode = 'human', observation_transform=StateTfm('cnn', n_frames=4), continuous=False, die_if_grass = True, lap_complete_percent = 1, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
    
    elif net_type == 'combine':
        env = ga.make(env_id, observation_transform=StateTfm('combine'), continuous=False, die_if_grass = die_if_grass, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        agent = DQN.DQN(
            name = 'COMBINED_DQN',
            policy = Combine_Policy(4, 7, 5, filters=[16, 32, 64], fcs=[256]).apply(utils.init_weights('kaiming')),
            env = env,
            action_space = list(range(5)),
            gamma=0.99,
            eps_start=1.0,
            eps_decay=0.995,
            eps_end=0.01,
            tau=1e-3,
            batch_size=64,
            update_every=4,
            device='auto',
            seed=seed
        ).to('cuda')

        env = ga.make(env_id, render_mode = 'human', observation_transform=StateTfm('combine', n_frames=4), continuous=False, die_if_grass = True, lap_complete_percent = 1, random_direction = False, render_ray=True)
        env.add_wrapper(SkipFrame, skip=4)
        

    agent.fit(n_games=100000, deterministic=False, save_best=True, save_every=100, save_dir=chkpt_dir, progress_bar=tqdm)
    agent.play(env, stop_if_truncated=True, seed = 720402)

if __name__ == "__main__":
    main()
