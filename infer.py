import infer_A2C
import infer_PPO
import infer_DQN

def main():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help'
    )

    parser.add_argument('--model', type=str, default='DQN', help='Model to use')
    parser.add_argument('--net_type', type=str, default='linear', help='Type of network to use')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    args = parser.parse_args()

    print(args.model)
    print(args.net_type)
    print(args.seed)

    model = args.model.lower()
    net_type = args.net_type.lower()

    if model == 'dqn':
        infer_DQN.play(net_type, args.seed)
    elif model == 'a2c':
        infer_A2C.play(net_type, args.seed)
    elif model == 'ppo':
        infer_PPO.play(net_type, args.seed)
    else:
        print("Invalid model")

if __name__ == "__main__":
    main()
