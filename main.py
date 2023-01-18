import time
import datetime
import os

from backdoored_fedavg3 import BackdooredFedAVG
from networks.BadNets import BadNetMNIST, CNN_RLR
from networks.MultilayerPerceptron import MLP
from networks.AlexNet import AlexNet
from networks.ResNet import ResNet18


def main():
    b = time.time()
    results = {'net': CNN_RLR,
               'nb_users': 100,
               'attack_target': 0,
               'attack_type': 'cross', #yellow_sticker
               'aggregation': 'avg',
               'label_skew': 1,
               'nb_attack': 20,
               'p_attack': 0.5,
               'dataset': 'MNIST', #FashionMNIST, GTSRB
               'data_path': "C:\\Users\\FE264791\\Documents\\4_FederatedLearning\\Data\\",
               #'data_path': "C:\\Users\\FE264791\\Documents\\4_FederatedLearning\\Data\\emnist-digits.mat",
               'rounds': 1,
               'E': 1,
               'B': 500,
               'C': 0.1,
               'lr': 0.01,
               'bd_freq': None,
               # 'None' for random selection of users, float 0-1 for freq of attack (/!\ when C=1 and nb_attack>0 and nb_malicious)
               'nb_malicious': 0,
               'viz': False,
               #'viz': ['fc2'],
               'freq_viz': 1,
               'nb_plots': 1,
               'sched_lr': False,
               'server_preload': None,
               #'server_preload': "C:\\Users\\FE264791\\Documents\\Code\\BackdoorFL\\results\\emoji\\Test1\\20221028_115851_Target5_AttackedUsers20_Frac0.5CNN_RLR\\Server_Model_round197best_CNN_RLR.pth",
               #'server_preload': "C:\\Users\\FE264791\\Documents\\Code\\BackdoorFL\\results\\bad_pattern_plus\\IID_Fashion\\20221228_171736_Target0_AttackedUsers20_Frac0.5CNN_RLR\\Server_Model_round190best_CNN_RLR.pth",
               'prune_layer': 'conv2',
               'eval': False,
               'cuda': True}

    bd_fedavg = BackdooredFedAVG(net=results['net'],
                                 nb_users=results['nb_users'],
                                 attack_target=results['attack_target'],
                                 attack_type=results['attack_type'],
                                 nb_attack=results['nb_attack'],
                                 p_attack=results['p_attack'],
                                 dataset=results['dataset'],
                                 data_path=results['data_path'],
                                 results_path=None,
                                 aggregation=results['aggregation'],
                                 label_skew=results['label_skew'],
                                 server_preload_net=results['server_preload'],
                                 cuda=results['cuda'])

    bd_fedavg.backdoored_fedvag(rounds=results['rounds'],
                                E=results['E'],
                                B=results['B'],
                                C=results['C'],
                                lr=0.01,
                                inter_viz=results['viz'],
                                save_net=False,
                                attack_rounds_freq=results['bd_freq'],
                                nb_malicious=results['nb_malicious'],
                                viz_rounds_freq=results['freq_viz'],
                                nb_plots=results['nb_plots'],
                                sched=results['sched_lr'],
                                eval_=results['eval'],

                                S=1,
                                sigma=6,
                                K=10)

    #bd_fedavg.server_pruning(results['prune_layer'], 'reverse_soft', plot=False, E=1, B=2, samples=6, lr=1, patience=3)
    bd_fedavg.build_protective_mask(bd_fedavg.server.net, plot=True, E_mask=1, B_mask=1, samples_mask=5, lr_mask=10, alpha=0.03,
                                    trigger_preload="C:\\Users\\FE264791\\Documents\\Code\\BackdoorFL\\results\\cross\\20230111_053434_Target7_AttackedUsers20_Frac0.5_MNIST__lbskew1_CNN_RLR\\common_trigger.pt")

    results['total_time'] = str(datetime.timedelta(seconds=time.time() - b))

    file = open(os.path.join(bd_fedavg.results_path, 'params.txt'), 'w')
    for key in results:
        file.write(key + ', ' + str(results[key]) + '\n')
    file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

