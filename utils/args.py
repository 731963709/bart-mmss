import argparse


def get_args():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--gpu', default=0, type=int, help="Use CUDA on the device.")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', default='train', type=str, help="Mode selection")
    parser.add_argument('--model_name', default='xxx', type=str, help="Mode selection")


    ## model set
    parser.add_argument('--n_patchs', default='49', type=int, help="the number of object")
    parser.add_argument('--n_prompt', default='8', type=int, help="the number of object")

    ## opti
    parser.add_argument('--opt', default='AdamW', type=str, help="optimize")
    parser.add_argument('--lr', default='5e-5', type=float, help="learning rate")
    parser.add_argument('--beta1', default='0.9', type=float, help="learning beta1")
    parser.add_argument('--beta2', default='0.998', type=float, help="learning beta2")
    parser.add_argument('--epoch', default='12', type=int, help="train epochs")

    ## dataset path
    parser.add_argument("--data_path", default='../MINE_data', type=str, help="")
    parser.add_argument("--logdir", default='../experiments_MINE', type=str, help="")
    parser.add_argument('--prename', default='default', type=str, help="")

    ## others
    parser.add_argument('--seed', default='1234', type=int, help="seed")
    parser.add_argument('--label_smoothing', default='0.1', type=float, help="")

    ## hyper-parameter
    parser.add_argument('--sim_temp', default='1.0', type=float, help="")
    parser.add_argument('--sim_alpha', default='0.5', type=float, help="")
    parser.add_argument('--ce2_beta', default='0.5', type=float, help="")
    parser.add_argument('--theta', default='0.2', type=float, help="")

    ##
    parser.add_argument('--use_kqv', default='True', type=str, help="")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print(get_args())

# CUDA_VISIBLE_DEVICES=1 python utils/args.py --batch-size 4