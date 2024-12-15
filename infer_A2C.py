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


import A2C


class Actor(nn.Module):
    def __init__(self, n_inp, n_actions, features = [256, 256]):
        super().__init__()

        layer_sizes = [n_inp] + features

        self.encoded = nn.Sequential()

        for i in range(len(layer_sizes) - 1):
            self.encoded.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.encoded.append(nn.ReLU(inplace=True))
        
        self.actor = nn.Linear(layer_sizes[-1], n_actions)

    def forward(self, x):
        return self.actor(self.encoded(x))

class Critic(nn.Module):
    def __init__(self, n_inp, features = [256, 256]):
        super().__init__()
        layer_sizes = [n_inp] + features

        self.encoded = nn.Sequential()

        for i in range(len(layer_sizes) - 1):
            self.encoded.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.encoded.append(nn.ReLU(inplace=True))
        
        self.critic = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x):
        return self.critic(self.encoded(x))
    
class Policy(nn.Module):
    def __init__(self, n_inp, n_actions: int, features: list[int] = [256, 256], optimizer = optim.Adam, actor_lr = 5e-4, critic_lr=5e-4, optimizer_kwargs = None):
        super().__init__()

        self.actor = Actor(n_inp, n_actions, features)
        self.critic = Critic(n_inp, features)

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
            rays = observation['rays'].astype(np.float32) / 200
            vels = observation['vels'].astype(np.float32) / 150
            vels[vels < 0] = 0
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
    chkpt_dir = 'checkpoints/BetterCarRacing-v0/A2C'

    env_id = "BetterCarRacing-v0"

    seed = 110404

    die_if_grass = True

    if net_type == 'linear':
        env = ga.make_vec(env_id, num_envs=n_envs, continuous=False, die_if_grass = die_if_grass, random_direction=False, observation_transform=StateTfm())
        agent = A2C.A2C(
            policy = Policy(14, 5, [256, 256], actor_lr = 5e-5, critic_lr=1e-4).apply(utils.init_weights('kaiming')),
            env = env,
            gamma=0.99,
            device='auto',
            seed=seed
        )

        env = ga.make(env_id, render_mode = 'human', continuous=False, die_if_grass = die_if_grass, random_direction=False, lap_complete_percent = 1, render_ray=False)
        env.set_observation_transform(StateTfm())

    elif net_type == 'cnn':
        env = ga.make(env_id, observation_transform=StateTfm('cnn', n_frames=4), continuous=False, die_if_grass = die_if_grass, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        agent = A2C.A2C(
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
            name = 'IMAGE_VEL_DQN',
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

        env = ga.make(env_id, render_mode = 'human', observation_transform=StateTfm('combine', n_frames=4), continuous=False, die_if_grass = True, lap_complete_percent = 1, random_direction = False)
        env.add_wrapper(SkipFrame, skip=4)
        

    agent.load(chkpt_dir, 'best')

    agent.play(env, stop_if_truncated=True, seed = 720402)

if __name__ == "__main__":
    main()
