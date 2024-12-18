# Reinforcement Learning Agent for Automatic Car Driving

## Description
This project focuses on developing an agent using deep reinforcement learning algorithms to drive a car in the Better Car Racing environment—a customized version of the CaRacing environment from OpenAI's Gym library.

## Installation
Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/trinhminh11/BetterCarRacingRL.git
    ```
2. Navigate to the project directory:
    ```bash
    cd BetterCarRacingRL
    ```
3. Install the required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Inference
To perform inference, use the following command in the terminal:

```bash
python infer_{algorithm}.py --net_type={type_of_network}
```

- **`{algorithm}`**: The algorithm to use. Choose from `dqn`, `a2c`, or `ppo`.
- **`{type_of_network}`**: The type of neural network used. Options are `linear`, `cnn`, or `combine`.

### Training
To train the agent, use the following command in the terminal:

```bash
python train_{algorithm}.py --net_type={type_of_network}
```

- **`{algorithm}`**: The algorithm to train. Choose from `dqn`, `a2c`, or `ppo`.
- **`{type_of_network}`**: The type of neural network used. Options are `linear`, `cnn`, or `combine`.

#### PPO Training Hyperparameters
When training the PPO agent, you can adjust the hyperparameters by passing them as arguments in the terminal:

- `--iter`: Number of iterations per episode. Default: `10`.
- `--coef_crit`: Coefficient of value loss. Default: `0.5`.
- `--coef_entropy`: Coefficient of entropy loss. Default: `0.1`.
- `--epsilon`: Clip value in policy loss. Default: `0.2`.
- `--clip_vf`: Clip value in value loss. Default: `10`.
- `--gamma`: Discount factor to calculate returns. Default: `0.99`.
- `--lambda`: GAE lambda. Default: `0.95`.

### Checkpoints
Our checkpoints are stored in https://drive.google.com/drive/u/0/folders/13I0mJML9XxceiFH6wCoKmndTez8bsLZG

Create this below folder to save checkpoint:

```bash
checkpoints/BetterCarRacing-v0/
```

Checkpoints of each agent will be saved in 

```bash
checkpoints/BetterCarRacing-v0/{algorithm}
```

## Contact
For further information, contact:

- **Facebook**: [https://www.facebook.com/vang.trinh.710](https://www.facebook.com/vang.trinh.710)
- **GitHub**: [https://github.com/trinhminh11](https://github.com/trinhminh11)
