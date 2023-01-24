import argparse

# local imports
from fl_bandage import BackdooredFedAVG
from networks import fl_net
from actors import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('nb_users', type=int, help='Number of users to be created.')
parser.add_argument('attack_target', type=int, help='Attack target.')
parser.add_argument('attack_type', type=str, help="Type of attack. Supported options are 'square', 'cross' and "
                                                  "'copyright' for MNIST, FashionMNIST and FEMNIST datasets, and"
                                                  "'yellow sticker' for GTSRB.")
parser.add_argument('nb_attack', type=int, help='Number of malicious clients.')
parser.add_argument('p_attack', type=float, help='Proportion of poised local dataset for malicious clients.')
parser.add_argument('dataset', type=str, help="Dataset to be used. Supported options are 'MNIST', 'FashionMNIST', "
                                              "'FEMNIST' and 'GTSBR'.")
parser.add_argument('rounds', type=int, help='Number of FL rounds.')
parser.add_argument('E', type=int, help='Number of local epochs.')
parser.add_argument('B', type=int, help='Local batch size.')
parser.add_argument('lr', type=float, help='Local learning rate.')
parser.add_argument('C', type=float, help='Proportion of clients sampled at each round.')
parser.add_argument('samples', type=int, help='Number of random samples in FL-Bandage.')
parser.add_argument('iterations', type=int, help='Number of iterations for trigger estimation.')
parser.add_argument('lr_bandage', type=float, help='Learning rate in FL-Bandage.')
parser.add_argument('gamma', type=float, help='Proportion of estimated trigger to be kept (gamma in FL-Bandage).')
parser.add_argument('-label_skew', '--label_skew', type=int, default=False, help='If label skewed distribution (non-IID), number of '
                                                                 'labels per client.')
parser.add_argument('-kernel_size', '--kernel_size', type=int, default=1, help='Kernel size in FL-Bandage. '
                                                                               '1 -> 3x3 kernel, 2->5x5 kernel')
parser.add_argument('-data_path', '--data_path', type=str, default=os.getcwd(),
                    help="Path where dataset is stored or will be downloaded. For dataset 'FEMNIST', file"
                         " 'emnist-digits.mat' must be located in file.")

parser.add_argument('-server_preload_net', '--server_preload_net', type=str, default=None,
                    help='(Optional) Path to .pth file if global model already trained.')

parser.add_argument('--save', action='store_true', help='Save results (.pth) after FL training and after defense.'
                                                        ' Otherwise do not mention.')
parser.add_argument('--cuda', action='store_true', help='Use option to computed in GPU. Otherwise do not mention.')
args = parser.parse_args()


def main():

    bd_fedvag = BackdooredFedAVG(fl_net, args.nb_users, args.attack_target, args.attack_type, args.nb_attack,
                                 args.p_attack, args.dataset, args.data_path, args.label_skew, args.server_preload_net,
                                 args.cuda)
    attack_results = bd_fedvag.backdoored_fedvag(args.rounds, args.E, args.B, args.C, args.lr, args.save)
    if len(attack_results) == 2:
        print('No attack applied during training time. Best accuracy {} at round {}'.format(attack_results[0],
                                                                                            attack_results[1]))
    else:
        print('-------------------------------------------FL-Backdoor training----------------------------------------')
        print('After backdoor attack training, clean accuracy is {}, backdoor attack success rate is {} and poised'
              ' accuracy is {}'.format(attack_results[0], attack_results[1], attack_results[2]))
        print('-------------------------------------------FL-Bandage Defense------------------------------------------')
        defense_results = bd_fedvag.test_defense(args.samples, args.iterations, args.lr_bandage, args.gamma, args.save,
                                                 kernel_size=args.kernel_size)
        print('After FL-Bandafe defense, clean accuracy is {}, backdoor attack success rate is {} and poised accuracy '
              'is {}'.format(defense_results[0], defense_results[1], defense_results[2]))


if __name__ == '__main__':
    main()
