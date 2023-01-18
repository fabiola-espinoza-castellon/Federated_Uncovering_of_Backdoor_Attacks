import random
import sys
import os
import datetime
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader, sampler
import numpy as np
import torch.nn.utils.prune as prune

from tools.metrics import accuracy_no_grad, conf_no_grad, acc_conf_cm_no_grad, confusion_matrix_no_grad, confusion_matrix_no_grad_mask, acc_conf_no_grad, confusion_matrix_no_grad_mask_expand2
from actors.users_manager import UsersManager
from tools.plots import viz_activations, best_size
from tools.save_tools import add_to_dict, save_state_dict, save_as_txt, path_exists, append_to_txt
from actors.server import Server
from custom_datasets.random_data import RandomData, UniformSamePixelsData


class BackdooredFedAVG:

    def __init__(self, net, nb_users, attack_target, attack_type, nb_attack, p_attack, dataset, data_path=os.getcwd(),
                 results_path=None, aggregation='avg', label_skew=False, server_preload_net=None, user_pruning=False,
                 no_poise_class=[], cuda=False):
        self.users_manager = UsersManager(dataset, nb_users, data_path)
        self.net = net(in_chan=self.users_manager.c, in_d=self.users_manager.d, out_d=self.users_manager.o)
        self.server = Server(net(in_chan=self.users_manager.c, in_d=self.users_manager.d, out_d=self.users_manager.o))
        self.nb_users = nb_users
        if server_preload_net:
            print('Preloaded server net')
            self.server.net.load_state_dict(torch.load(server_preload_net))
        if cuda:
            self.server.net = self.server.net.cuda()
        self.attack_target = attack_target
        self.attack_type = attack_type
        self.nb_attack = nb_attack
        self.p_attack = p_attack
        self.aggregation = aggregation
        self.label_skew = label_skew
        self.rounds = 0
        if results_path:
            assert os.path.exists(results_path), ' Specified results folder does not exist.'
            self.results_path = results_path
        else:
            # Will create folder with actual date and time
            self.results_path = os.path.join(os.getcwd(), "results", self.aggregation, self.attack_type,
                                             datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                             + '_Target' + str(self.attack_target)
                                             + '_AttackedUsers'+str(self.nb_attack)
                                             + '_Frac' + str(self.p_attack)
                                             + '_' + dataset + '_'
                                             + '_lbskew' + str(self.label_skew) + '_'
                                             + type(self.server.net).__name__)
        self.user_pruning = user_pruning
        self.no_poise_class = no_poise_class
        self.cuda = cuda
        self.results = {'net': type(self.net).__name__, 'nb_users': nb_users, 'data_path': data_path,
                        'attack_target':attack_target, 'attack_type': attack_type, 'nb_attack': nb_attack,
                        'p_attack': p_attack, 'dataset': dataset, 'aggregation': aggregation, 'label_skew': label_skew,
                        'server_preload': server_preload_net, 'user_pruning': user_pruning,
                        'no_poise_class': no_poise_class, 'cuda': cuda}

    def initialize(self):
        # Fix seed
        # torch.manual_seed(0)
        # random.seed(0)

        if self.label_skew:
            self.users_manager.create_users_label_skew(self.label_skew)
        else:
            self.users_manager.create_users()
        if self.attack_type == 'no_attack':
            self.nb_attack = 0
        else:
            self.users_manager.poise_users(self.attack_target, self.nb_attack, self.p_attack, pattern=self.attack_type,
                                           no_poise_class=self.no_poise_class)
        return None

    def user_update(self, user, E, B, lr, scheduler=False, eval_=False, viz_layers=False, save_net=False, **kwargs):
        user_net = self.net
        user_net.load_state_dict(self.server.net.state_dict())
        if self.cuda:
            user_net = user_net.cuda()
        samples = sampler.SubsetRandomSampler(user.train_indexes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(user_net.parameters(), lr=lr)
        if scheduler:
            # scheduler_ = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=int(E/4))
            # Mc Mahan papers actually multiplies lr by 0.99 per round
            optimizer = torch.optim.SGD(user_net.parameters(), lr=lr*(0.99**np.floor(self.rounds/2)))

        for epoch in range(E):
            dataloader_train = DataLoader(self.users_manager.train_samples, batch_size=B, sampler=samples,
                                          shuffle=False, num_workers=0)
            for d, t in dataloader_train:
                if self.cuda:
                    d, t = d.cuda(), t.cuda()
                optimizer.zero_grad()
                output = user_net(d)
                loss = criterion(output, t)
                loss.backward()
            #if scheduler:
                #scheduler_.step(loss)
                #optimizer.step()
            #else:
                optimizer.step()

        if eval_:
            # Not suitable for non-iif cases, only to evaluate attacks because testing on all clean or poised dataset.
            self.users_manager.users_accuracy_clean[user.idx][self.rounds-1] = accuracy_no_grad(
                self.users_manager.test_samples, user_net, samp=self.data_sampler('clean'), cuda=self.cuda)

            self.users_manager.users_accuracy_poise[user.idx][self.rounds-1] = accuracy_no_grad(
                self.users_manager.test_samples, user_net, samp=self.data_sampler('poised'), cuda=self.cuda)

        if isinstance(kwargs.get('viz_layers', viz_layers), list):
            # Visualize activations and weights (FC layers)
            self.viz_delta_inter_activations(layers=kwargs['inter_viz'], network=user_net, name='{}User{}_{}'.format(
                                           ['Malicious' if user.attacked else 'Loyal'][0], user.idx, type(user_net).__name__))
            viz_activations(user_net.state_dict()[kwargs['inter_viz'][0]+'.weight'], name=os.path.join(
                self.results_path,'viz_round{}_{}User{}_weight{}_{}.png'.format(self.rounds, ['Malicious' if user.attacked else 'Loyal'][0],
                user.idx, kwargs['inter_viz'][0], type(user_net).__name__)))

        if save_net:
            if user.attacked:
                n = 'malic'
            else:
                n = 'loyal'

            if self.user_pruning:
                # Pruning per user
                self.pruning(user_net, self.user_pruning, type_='spectral_des', name='{}User{}'.format(
                                           ['Malicious' if user.attacked else 'Loyal'][0],
                                           user.idx), projection=False)

            save_state_dict(user_net, 'user{}{}_round{}_{}'.format(user.idx, n, self.rounds, type(user_net).__name__),
                            self.results_path)
        return user_net.state_dict()
        #return {k: self.server.net.state_dict()[k] - user_net.state_dict()[k] for k in user_net.state_dict().keys()}

    def backdoored_fedvag(self, rounds, E, B, C=0.1, lr=0.01, sched=False, save_net=False, freq_attack=float('inf'),
                          freq_viz_layers=float('inf'), eval_=False, **kwargs):
        self.results = dict(self.results,
                            **{'rounds': rounds, 'E_bd': E, 'B_bd': B, 'C': C, 'lr_bd': lr, 'sched': sched,
                               'save_net': save_net, 'freq_attack': freq_attack, 'freq_viz_layers': freq_viz_layers,
                               'eval_': eval_})
        self.results.update(kwargs)

        # Assertions
        if freq_attack != float('inf'):
            assert 'nb_malicious' in kwargs, 'Specify nb_malicious in kwargs.'
        if freq_viz_layers != float('inf'):
            assert 'nb_plots' and 'viz_layers' in kwargs, \
                'Specify the frequence of user visualization, the number of plots and the layers.'
        if kwargs.get('pruning_freq', float('inf')) < float('inf'): # Pruning freq must be < inf to prune
            assert 'prune_layer' and 'pruning_type' in kwargs, 'Specify type of pruning and layer.'

        self.initialize()

        best_round, best_acc = 0, 0

        # create matrix to evaluate users
        if eval_:
            self.users_manager.users_accuracy_clean = [[float('nan')] * rounds for _ in range(self.nb_users)]
            self.users_manager.users_accuracy_poise = [[float('nan')] * rounds for _ in range(self.nb_users)]

        # Save init model
        # save_state_dict(self.server.net, 'Server_Model_round0_{}'.format(type(self.server.net).__name__),
        # path_exists(self.results_path))

        clean_accs, poised_accs = [], []

        # From 1 to round+1 for freqs
        for round_ in range(1, rounds+1):
            print('Round {}/{}'.format(round_, rounds))
            self.rounds = round_
            N = 0
            #clean_acc = self.test_server(type_='clean')
            #clean_conf = self.conf_server(type_='clean')
            clean_acc = self.conf_server(type_='clean', loss=True)
            clean_accs += [clean_acc]
            if self.users_manager.poised_train_indexes:
                # poised_acc = self.test_server(type_='poised')
                # poised_conf = self.conf_server(type_='poised')
                poised_acc = self.conf_server(type_='poised', loss=True)
                poised_accs += [poised_acc]

            # Remember round of best accuracy : will be associated to accuracy before training
            if clean_acc > best_acc:
                best_acc, best_round = clean_acc, round_
                if save_net:
                    save_state_dict(self.server.net, 'server_net', path_exists(self.results_path))

            if freq_attack != float('inf'):
                if round_ % freq_attack == freq_attack - 1:
                    h = kwargs['nb_malicious']
                # loyal training for other rounds
                else:
                    h = 0
            else:
                h = 'random'

            if round_ % freq_viz_layers == freq_viz_layers - 1:
                bd_plots, clean_plots = 0, 0

            sampled = self.select_users(C=C, nb_attacks=h)
            kwargs['K'] = int(len(sampled))
            for user in sampled:
                save_user_net = False
                if round_ % freq_viz_layers == freq_viz_layers - 1:
                    if user.attacked and bd_plots < kwargs['nb_plots']:
                        bd_plots += 1
                    elif not user.attacked and clean_plots < kwargs['nb_plots']:
                        clean_plots += 1
                    save_user_net = True

                params = self.user_update(user, E, B, lr, scheduler=sched, eval_=eval_, save_net=save_user_net, **kwargs)
                N += user.nk

                # Send to server
                self.server.receive_from_user(params, user.nk, aggregation=self.aggregation, **kwargs)
            #self.server.check_weigths()
            if self.aggregation == 'dp':
                N = kwargs['K']
            self.server.update_average_weights(N=N, aggregation=self.aggregation, krum_f=int(self.nb_attack*C))

            if round_ % freq_viz_layers == freq_viz_layers - 1:
                self.viz_delta_inter_activations(layers=kwargs['viz_layers'], network=self.server.net,
                                                 name='Server_'+str(type(self.server.net).__name__))

            # Save network before pruning
            #if round_ == kwargs.get('pruning_begin'):
            #    save_state_dict(self.server.net, 'Server_bf_pruning_round{}'.format(round_), path_exists(self.results_path))

            if round_ % kwargs.get('pruning_freq', float('inf')) == kwargs.get('pruning_freq', float('inf')) - 1 and \
                    round_ > kwargs.get('pruning_begin'): #attack already efficient
                self.server_pruning(kwargs['prune_layer'], kwargs['pruning_type'], plot=True, E=5, B=1, samples=1, lr=1, patience=8)

        # Plot accs
        if clean_accs:
            plt.plot(clean_accs, label='Clean acc')
            plt.plot(poised_accs, label='Bd success')
            plt.xlabel('Rounds')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(path_exists(self.results_path), 'viz_Server_clean_poised_acc.png'))
            plt.close()

        # Rename best net
        if save_net and os.path.exists(os.path.join(self.results_path, 'server_net.pth')):
            # Keep best net
            self.server.net.load_state_dict(torch.load(os.path.join(path_exists(self.results_path), 'server_net.pth')))
            os.rename(os.path.join(path_exists(self.results_path), 'server_net.pth'),
                      os.path.join(path_exists(self.results_path), 'Server_Model_round{}best_{}.pth'.
                                   format(best_round, type(self.server.net).__name__)))

        if eval_:
            save_as_txt(self.users_manager.users_accuracy_clean,
                        'User_cleanAccuracy_round{}_B{}_E{}'.format(self.rounds, B, E),
                        path_exists(self.results_path))
            save_as_txt(self.users_manager.users_accuracy_poise,
                        'User_poiseAccuracy_round{}_B{}_E{}'.format(self.rounds, B, E),
                        path_exists(self.results_path))

        return None

    def select_users(self, C=0.1, nb_attacks='random'):
        total_nb_users = len(self.users_manager)
        K = max(1, round(C * total_nb_users))

        if nb_attacks == 'random':
            # Randomly select users (can be attackers or loyal)
            return random.sample(self.users_manager.users, k=K)
        elif isinstance(nb_attacks, int):
            # Select nb_attacks malicious clients and rest loyal
            # 0 for only loyal clients
            assert nb_attacks < K, 'Can not select more malicious users than size of sample.'
            loyal = random.sample([u for u in self.users_manager.users if u.attacked is False], k=K - nb_attacks)
            malic = random.sample([u for u in self.users_manager.users if u.attacked is True], k=nb_attacks)
            return loyal + malic
        else:
            pass

    def data_sampler(self, type_='clean'):
        if type_ == 'clean':
            return sampler.SubsetRandomSampler(self.users_manager.clean_test_indexes)
        elif type_ == 'poised':
            return sampler.SubsetRandomSampler(self.users_manager.poised_test_indexes)
        elif isinstance(type_, list):
            return sampler.SubsetRandomSampler(type_)
        else:
            sys.exit('Chose which data to sample.')

    def test_server(self, type_='clean'):
        if type_ == 'clean':
            acc = accuracy_no_grad(self.users_manager.test_samples, self.server.net, samp=self.data_sampler(type_),
                                   cuda=self.cuda)
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_cleanAcc_{}.txt'.format(type(self.server.net).__name__)))
            #self.server.clean_accuracy_per_round += [acc]

        elif type_ == 'poised':
            acc = accuracy_no_grad(self.users_manager.test_samples, self.server.net, samp=self.data_sampler(type_),
                                   cuda=self.cuda)
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_bdSuccess_{}.txt'.format(type(self.server.net).__name__)))
            #self.server.bd_success_per_round += [acc]

        else:
            sys.exit('Chose which data to test on.')

        return acc

    def conf_server(self, type_='clean', loss=False):
        if type_ == 'clean':
            if loss:
                acc, conf, l = acc_conf_no_grad(self.users_manager.test_samples, self.server.net,
                                                samp=self.data_sampler(type_), loss_val=loss, cuda=self.cuda)
            else:
                acc, conf = acc_conf_no_grad(self.users_manager.test_samples, self.server.net,
                                             samp=self.data_sampler(type_), cuda=self.cuda)
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_cleanAcc_{}.txt'.format(type(self.server.net).__name__)))
            append_to_txt(conf, os.path.join(path_exists(self.results_path),
                                             'Server_cleanConf_{}.txt'.format(type(self.server.net).__name__)))
            if loss:
                append_to_txt(l, os.path.join(path_exists(self.results_path), 'Server_clean_loss.txt'))
            return acc

        elif type_ == 'poised':
            if loss:
                acc, conf, l = acc_conf_no_grad(self.users_manager.test_samples, self.server.net,
                                                samp=self.data_sampler(type_), loss_val=loss, cuda=self.cuda,
                                                target_label=self.attack_target)
            else:
                acc, conf = acc_conf_no_grad(self.users_manager.test_samples, self.server.net,
                                             samp=self.data_sampler(type_), cuda=self.cuda,
                                             target_label=self.attack_target)
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_bdSuccess_{}.txt'.format(type(self.server.net).__name__)))
            append_to_txt(conf, os.path.join(path_exists(self.results_path),
                                             'Server_bdConf_{}.txt'.format(type(self.server.net).__name__)))
            if loss:
                append_to_txt(l, os.path.join(path_exists(self.results_path), 'Server_bd_loss.txt'))
            return acc

        else:
            sys.exit('Chose which data to test on.')

    def acc_conf_server(self, type_='clean', loss=False):
        #acc, conf = acc_conf_no_grad(self.users_manager.test_samples, self.server.net, samp=self.data_sampler(type_),
        #                             cuda=self.cuda)
        if not loss:
            acc, conf, conf_mean, cm = acc_conf_cm_no_grad(self.users_manager.test_samples, self.server.net,
                                                           samp=self.data_sampler(type_), cuda=self.cuda)
        else:
            if type_ == 'poised':
                acc, conf, conf_mean, cm, losses = acc_conf_cm_no_grad(self.users_manager.test_samples, self.server.net,
                                                                       samp=self.data_sampler(type_), cuda=self.cuda,
                                                                       loss_val=loss,
                                                                       target_label=self.attack_target)
            else:
                acc, conf, conf_mean, cm, losses = acc_conf_cm_no_grad(self.users_manager.test_samples, self.server.net,
                                                           samp=self.data_sampler(type_), cuda=self.cuda, loss_val=loss)

        if type_ == 'clean':
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_cleanAcc_{}.txt'.format(type(self.server.net).__name__)))
            append_to_txt(conf_mean, os.path.join(path_exists(self.results_path),
                                            'Server_cleanConf_{}.txt'.format(type(self.server.net).__name__)))
            save_as_txt(conf, 'Server_cleanConf_perTarget', path_exists(self.results_path))
            save_as_txt(cm, 'Server_cleanCM', path_exists(self.results_path))
            if loss:
                append_to_txt(losses, os.path.join(path_exists(self.results_path), 'Server_clean_loss.txt'))

        elif type_ == 'poised':
            # BD success is the images predicted as the target class
            acc = np.sum(cm[:, self.attack_target])/np.sum(cm)
            append_to_txt(acc, os.path.join(path_exists(self.results_path),
                                            'Server_bdSuccess_{}.txt'.format(type(self.server.net).__name__)))
            append_to_txt(conf_mean, os.path.join(path_exists(self.results_path),
                                            'Server_bdConf_{}.txt'.format(type(self.server.net).__name__)))
            save_as_txt(conf, 'Server_bdConf_perTarget', path_exists(self.results_path))
            save_as_txt(cm, 'Server_bdCM', path_exists(self.results_path))
            if loss:
                append_to_txt(losses, os.path.join(path_exists(self.results_path), 'Server_bd_loss.txt'))

        else:
            sys.exit('Chose which data to test on.')

        return acc, conf_mean

    def inter_activations(self, layers, network, type_='clean'):
        if type_ in ['clean', 'poised'] and isinstance(layers, str):
            if type_+'_'+layers not in self.server.activations.keys():
                dataloader = DataLoader(self.users_manager.test_samples, batch_size=64,
                                        sampler=self.data_sampler(type_),
                                        shuffle=False, num_workers=0)
                intermediate_outputs = []
                with torch.no_grad():
                    network.eval()
                    for d, t in dataloader:
                        if self.cuda:
                            d = d.cuda()
                        intermediate_outputs += [network.intermediate_forward(layers, d).detach().cpu()]
                    intermediate_outputs = torch.mean(torch.vstack(intermediate_outputs), axis=0)
                self.server.activations[type_+'_'+layers] = intermediate_outputs
                return self.server.activations[type_+'_'+layers]
            else:
                return self.server.activations[type_+'_'+layers]
        else:
            dataloader = DataLoader(self.users_manager.test_samples, batch_size=64, sampler=self.data_sampler(type_),
                                    shuffle=False, num_workers=0)
            intermediate_outputs = {}
            for layer in layers:
                intermediate_outputs[layer] = []
                with torch.no_grad():
                    network.eval()
                    for d, t in dataloader:
                        if self.cuda:
                            d = d.cuda()
                        intermediate = network.intermediate_forward(layer, d)
                        intermediate_outputs[layer] += [intermediate.detach().cpu()]
                    intermediate_outputs[layer] = torch.mean(torch.vstack(intermediate_outputs[layer]), axis=0)
            return intermediate_outputs

    def viz_inter_activations(self, layers, network, type_, name):
        inter = self.inter_activations(layers, network, type_=type_)
        for l in layers:
            '''
            if len(inter[l].shape) == 1:
                order = torch.argsort(inter[l]).tolist()
            elif len(inter[l].shape) > 1:
                order = torch.argsort(torch.sum(torch.abs(inter[l]), dim=[1, 2])).tolist()
            '''
            viz_activations(inter[l],
                            title='{} activations'
                            #      ' \n{}\n{}'
                            .format(l,
                            #self.attack_target, order[:len(order)//2],
                            #order[len(order)//2:]
                            ),
                            name=os.path.join(path_exists(self.results_path), name + '_' + l + '.png'))
        return None

    def viz_delta_inter_activations(self, layers, network, name):
        clean = self.inter_activations(layers, network, type_='clean')
        for l in layers:
            #order = torch.argsort(torch.sum(torch.abs(clean[l]), dim=[1, 2])).tolist()
            viz_activations(clean[l],
                            title='Clean {} activations '
                                  #'\n{}'
                                  #'\n{}'
                            .format(l,
                                    #order[:len(order) // 2],
                                    #order[len(order) // 2:]
                                    ),
                            name=os.path.join(path_exists(self.results_path), 'viz_round{}_Clean'.format(self.rounds) + name + l + '.png'))

        if len(self.users_manager.poised_train_indexes) > 0:
            poised = self.inter_activations(layers, network, type_='poised')
            delta = {k: torch.abs(clean[k] - poised[k]) for k in clean.keys()}
            for l in layers:
                #order = torch.argsort(torch.sum(torch.abs(poised[l]), dim=[1, 2])).tolist()
                viz_activations(poised[l],
                                title='Poised {} activations'
                                      #'\n{}'
                                      #'\n{}'
                                .format(l,
                                        #order[:len(order) // 2],
                                        #order[len(order) // 2:]
                                        ),
                                name=os.path.join(path_exists(self.results_path), 'viz_round{}_Poised'.format(self.rounds) + name + l + '.png'))

                #order = torch.argsort(torch.sum(torch.abs(delta[l]), dim=[1, 2])).tolist()
                viz_activations(delta[l],
                                title='Delta {} activations'
                                      #'\n{}'
                                      #'\n{}'
                                .format(l,
                                        #order[:len(order) // 2],
                                        #order[len(order) // 2:]
                                        ),
                                name=os.path.join(path_exists(self.results_path), 'viz_round{}_Delta'.format(self.rounds) + name + l + '.png'))
        else:
            pass
        return None

    def pruning(self, net, layer, type_='L1', name='', projection=False, plot=False, lambda_=0.001, **kwargs):  #one layer for now
        w = net.state_dict()
        f = None
        pruning_acc_clean = []
        pruning_acc_poise = []

        if type_ == 'weight':
            dims = net.state_dict()[layer + '.weight'].shape
            f = torch.argsort(torch.sum(torch.abs(net.state_dict()[layer + '.weight']), dim=list(range(len(dims))[1:])),
                              descending=True)

        elif type_ == 'bias':
            dims = net.state_dict()[layer + '.bias'].shape
            if len(dims) > 1:
                f = torch.argsort(torch.sum(torch.abs(net.state_dict()[layer + '.bias']),
                                            dim=list(range(len(dims))[1:])), descending=True)
            else:
                f = torch.argsort(torch.abs(net.state_dict()[layer + '.bias']), descending=True)

        elif type_ == 'L1': #can't use L1 if not on server
            dims = self.inter_activations(layer, net, 'clean').shape
            if len(dims) > 1:
                f = torch.argsort(torch.sum(torch.abs(self.server.activations['clean_'+layer]), dim=[1, 2]), descending=False)
            else:
                f = torch.argsort(torch.abs(self.server.activations['clean_'+layer]), descending=False)

        elif type_ == 'L1_poised':
            dims = self.inter_activations(layer, net, 'poised').shape
            if len(dims) > 1:
                f = torch.argsort(torch.sum(torch.abs(self.server.activations['poised_'+layer]), dim=[1, 2]), descending=True)
            else:
                f = torch.argsort(torch.abs(self.server.activations['poised_'+layer]), descending=True)

        elif 'spectral' in type_:
            # for now only on 2d matrix
            weights = net.state_dict()[layer + '.weight'].detach()
            # Trying to prune on other than fc
            if len(weights.shape)>2:
                weights = torch.mean(weights, axis=[2,3])

            weights = weights.T - weights.mean(axis=1)
            C = torch.cov(weights, correction=0) #rows: vars, column: observation
            u, s, v = torch.linalg.svd(C, full_matrices=True)
            if type_ == 'spectral_des':
                #f = argsort(dot(C, v[0, :]))[::-1] #descending
                # Je dois avoir 128 valeurs a trier parce que c'est la dim de mes observations
                f = torch.argsort(torch.abs(torch.matmul(weights.T, v[0, :])), descending=True)  # descending
                if projection:
                    xs, ys = torch.matmul(weights.T, v[0, :]), torch.matmul(weights.T, v[1, :])
                    fig, ax = plt.subplots()
                    ax.scatter(xs.cpu().numpy(), ys.cpu().numpy(), marker='+')
                    ax.vlines(xs[20].cpu().numpy(), ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='red',
                              linestyles='dashed')
                    ax.spines['left'].set_position('zero')
                    ax.spines['right'].set_color('none')
                    ax.spines['bottom'].set_position('zero')
                    ax.spines['top'].set_color('none')

                    # remove the ticks from the top and right edges
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')

                    plt.savefig(os.path.join(path_exists(self.results_path), '{}_projection_{}.png'.format(name, type_)), format='png',
                                bbox_inches='tight')
                    plt.close()

                    save_as_txt(xs.cpu().numpy(), '{}_projection_xs_{}.png'.format(name, type_), path_exists(self.results_path))
                    save_as_txt(ys.cpu().numpy(), '{}_projection_ys_{}.png'.format(name, type_), path_exists(self.results_path))
                    save_as_txt(torch.argsort(xs, descending=True).cpu().numpy(), '{}_sorted_weights'.format(name),
                                path_exists(self.results_path))

            else:
                f = torch.argsort(torch.abs(torch.matmul(weights.T, v[0, :])), descending=False)  # ascending
                #f = argsort(dot(C, v[0, :]))

        elif type_ == 'random_diff':
            r = int(len(self.users_manager.test_samples)/64)
            intermediate_outputs = []
            delta_intermediate_outputs = []
            # To Do : case where  more than 1 channel
            #l, L = self.users_manager.test_samples.data.shape[:-3]
            for i in range(r):
                rand = torch.rand(64, 1, 28, 28)
                delta_rand = rand + 0.5 * rand
                with torch.no_grad():
                    net.eval()
                    if self.cuda:
                        rand = rand.cuda()
                        delta_rand = delta_rand.cuda()
                    intermediate_outputs += [self.server.net.intermediate_forward(layer, rand).detach()]
                    delta_intermediate_outputs += [self.server.net.intermediate_forward(layer, delta_rand).detach()]
            intermediate_outputs = torch.mean(torch.vstack(intermediate_outputs), axis=0)
            delta_intermediate_outputs = torch.mean(torch.vstack(delta_intermediate_outputs), axis=0)
            f = torch.argsort(torch.abs(intermediate_outputs - delta_intermediate_outputs), descending=True)

        elif type_ == 'unif_diff':
            r = int(len(self.users_manager.test_samples)/64)
            intermediate_outputs = []
            delta_intermediate_outputs = []
            # To Do : case where  more than 1 channel
            #l, L = self.users_manager.test_samples.data.shape[:-3]
            for i in range(r):
                unif = 0.5*torch.ones((64, 1, 28, 28))
                delta_unif = unif + 0.5 * unif
                with torch.no_grad():
                    net.eval()
                    if self.cuda:
                        unif = unif.cuda()
                        delta_unif = delta_unif.cuda()
                    intermediate_outputs += [self.server.net.intermediate_forward(layer, unif).detach()]
                    delta_intermediate_outputs += [self.server.net.intermediate_forward(layer, delta_unif).detach()]
            intermediate_outputs = torch.mean(torch.vstack(intermediate_outputs), axis=0)
            delta_intermediate_outputs = torch.mean(torch.vstack(delta_intermediate_outputs), axis=0)
            f = torch.argsort(torch.abs(intermediate_outputs - delta_intermediate_outputs), descending=True)

        elif type_ == 'df_lip':
            f = torch.norm(net.state_dict()[layer + '.weight'].detach(), p=2, dim=1)
            f = torch.argsort(f, descending=True)

        elif type_ == 'L1_mask':
            if self.cuda:
                self.users_manager.pattern = self.users_manager.pattern.cuda()
            i = net.intermediate_forward(layer, self.users_manager.pattern.reshape(1, 1, 28, 28)).detach()
            f = torch.argsort(torch.abs(i.squeeze()), descending=True)

        elif 'reverse' in type_:
            # Freeze weights
            for param in net.parameters():
                param.requires_grad = False

            # shape = [len(self.users_manager.test_samples)]+list(self.users_manager.test_samples[0][0].shape)
            shape = [kwargs['samples']] + list(self.users_manager.test_samples[0][0].shape)
            E = kwargs['E']
            B = kwargs['B']
            lr = kwargs['lr']

            # train_samples = torch.rand(shape)
            seed = 0
            train_samples = UniformSamePixelsData(shape, 0, seed=seed)

            if plot:
                torch.save(train_samples.data, os.path.join(path_exists(self.results_path),
                                                            'train_samples_beginning.pt'))
                fig, axs = plt.subplots()
                trigger = torch.mean(train_samples.data, dim=0).squeeze().detach()
                vmin, vmax = torch.min(trigger), torch.max(trigger)
                if len(trigger.shape) > 2:
                    trigger = torch.permute(trigger, (1, 2, 0))
                    color = 'viridis'
                else:
                    color = 'gray'

                im = axs.imshow(trigger, interpolation=None, cmap=color, vmin=vmin, vmax=vmax)
                fig.colorbar(im, shrink=0.90)
                plt.savefig(os.path.join(path_exists(self.results_path), 'begin_reverse_trigger.png'), format='png',
                            bbox_inches='tight')
                plt.close()

            loss_dict = {int(k): np.zeros(E) for k in self.users_manager.test_samples.class_to_idx.values()}
            criterion = torch.nn.CrossEntropyLoss()
            min_value = float('inf')
            min_target = 0

            for target in loss_dict.keys():
                train_samples_target = UniformSamePixelsData(shape, target, seed=seed)
                train_samples_target.data.requires_grad = True
                optimizer = torch.optim.SGD([train_samples_target.data], lr=lr)
                dataloader = DataLoader(train_samples_target, batch_size=B, shuffle=False)
                for epoch in range(E):
                    for d, t in dataloader:
                        optimizer.zero_grad()
                        if self.cuda:
                            d, t = d.cuda(), t.cuda()
                        output = net(d)
                        loss = criterion(output, t)
                        if 'reg' in type_:
                            loss = loss + lambda_ * torch.norm(d, 1)
                            type_ += str(lambda_).replace('.', '')
                        loss_dict[target][epoch] += loss.item()
                        loss.backward()
                        optimizer.step()

                    loss_dict[target][epoch] /= len(dataloader)

                with open(os.path.join(path_exists(self.results_path), 'reverse_trigger_round{}_label{}.npy'.
                                 format(self.rounds, target)), 'wb') as file:
                    np.save(file, train_samples_target.data.detach().numpy())
                m = torch.mean(train_samples_target.data, dim=0).squeeze().detach()
                if len(m.shape) > 2:
                    m = torch.permute(m, (1, 2, 0))
                    color = 'viridis'
                else:
                    color = 'gray'
                fig, axs = plt.subplots()
                im = axs.imshow(m, interpolation=None, cmap=color, vmin=torch.min(m), vmax=torch.max(m))
                fig.colorbar(im, shrink=0.90)
                plt.savefig(
                    os.path.join(path_exists(self.results_path), 'reverse_trigger_round{}_label{}.png'.
                                 format(self.rounds, target)), format='png', bbox_inches='tight')
                plt.close()

                if loss_dict[target][-1] < min_value:
                    min_target = target
                    min_value = loss_dict[target][-1]
                    train_samples = train_samples_target

                plt.plot(loss_dict[target], label=str(target))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(path_exists(self.results_path), 'viz_trigger_loss.png'), format='png')
            plt.close()
            save_as_txt(list(loss_dict.values()), 'trigger_loss_per_target',
                        os.path.join(path_exists(self.results_path)))

            # Enhanced triggers
            means, stds = train_samples.data.mean(dim=[1, 2, 3], keepdim=True), train_samples.data.std(dim=[1, 2, 3], keepdim=True)
            train_samples.data = torch.where(torch.abs(train_samples.data-means) < stds,
                                             torch.zeros_like(train_samples.data), train_samples.data)

            if plot:
                fig, axs = plt.subplots()
                # trigger = train_samples.squeeze().detach()
                with open(os.path.join(path_exists(self.results_path), 'reverse_trigger.npy'), 'wb') as file:
                    np.save(file, train_samples.data.detach().numpy())
                viz_activations(train_samples.data.squeeze().detach(),
                                name=os.path.join(self.results_path, 'reverse_triggers_round{}.png'.format(self.rounds)), colors='gray')
                trigger = torch.mean(train_samples.data, dim=0).squeeze().detach()
                vmin, vmax = torch.min(trigger), torch.max(trigger)
                if len(trigger.shape) > 2:
                    trigger = torch.permute(trigger, (1, 2, 0))
                    color = 'viridis'
                else:
                    color = 'gray'
                im = axs.imshow(trigger, interpolation=None, cmap=color, vmin=vmin, vmax=vmax)
                fig.colorbar(im, shrink=0.90)
                plt.savefig(os.path.join(path_exists(self.results_path), 'reverse_trigger_round{}.png'.format(self.rounds)), format='png',
                            bbox_inches='tight')
                plt.close()

            with torch.no_grad():
                net.eval()
                if self.cuda:
                    train_samples.data = train_samples.data.cuda()
                output = self.server.net.intermediate_forward(layer, train_samples.data).detach()
            # f = torch.argsort(torch.abs(output.squeeze()), descending=True)
            if len(output.shape) > 3: #conv layer
                f = torch.argsort(torch.sum(torch.abs(torch.mean(output, dim=0)), axis=[1,2]), descending=True)
            else: #fc layers
                f = torch.argsort(torch.abs(torch.mean(output, dim=0)), descending=True)

        else:
            exit('Criteria not implemented.')

        mask_ = torch.ones(net.state_dict()[layer+'.weight'].shape).cuda()
        # loss_dict = {int(k): zeros(mask_.shape[0]) for k in self.users_manager.test_samples.class_to_idx.values()}
        loss_val = []
        bd_success = []
        best_bd = float('inf')
        best_index = 0
        append_to_txt(f, os.path.join(path_exists(self.results_path), 'pruning_filters_round'))
        # for j in range(int(len(f)/2)):
        """
        for i, j in enumerate(f[:int(len(f)/2)]): #Only prune half of layers
            if 'soft' in type_:
                mask_[j] = 0.5
                with torch.no_grad():
                    getattr(getattr(net, layer), 'weight')[:] *= mask_
            else:
                # indexes = f[2*j:2*j+2] mask_[indexes] = 0
                mask_[j] = 0

                # Naive solution for ResNet
                if '.' in layer:
                    layer = layer.split('.')
                    prune.custom_from_mask(getattr(getattr(net, layer[0])[int(layer[1])], layer[2]), name='weight', mask=mask_)
                else:
                    prune.custom_from_mask(getattr(net, layer), name='weight', mask=mask_)

            if 'reverse' in type_:
                criterion = torch.nn.CrossEntropyLoss()
                #for target in loss_dict.keys(): targets = target*torch.ones(len(train_samples.data), dtype=torch.long)
                targets = min_target * torch.ones(len(train_samples.data), dtype=torch.long)
                if self.cuda:
                    train_samples.data, targets = train_samples.data.cuda(), targets.cuda()
                output = net(train_samples.data)
                bd_success += [torch.sum(torch.argmax(output, 1) == targets).item()/len(targets)]
                loss_val += [criterion(output, targets).item()]
                # loss_dict[target][i] += loss.item()

                if bd_success[i] < best_bd:
                    best_index, best_bd = i, bd_success[i]
                    #break
            acc_c, cm_c = confusion_matrix_no_grad(self.users_manager.test_samples, net, samp=self.data_sampler('clean'),
                                               cuda=self.cuda)
            pruning_acc_clean += [acc_c]
            _, cm_p = confusion_matrix_no_grad(self.users_manager.test_samples, net, samp=self.data_sampler('poised'),
                                   cuda=self.cuda)
            pruning_acc_poise += [np.sum(cm_p[:, self.attack_target])/np.sum(cm_p)]
            append_to_txt(pruning_acc_clean[-1], os.path.join(path_exists(self.results_path), '{}_pruning_acc_clean_{}'.format(name, type_)))
            append_to_txt(pruning_acc_poise[-1], os.path.join(path_exists(self.results_path), '{}_pruning_acc_poise_{}'.format(name, type_)))

        # Plot
        if 'reverse' in type_:
            abs_diff = np.abs(np.array(pruning_acc_poise) - np.array(pruning_acc_clean))
            print('Estimated target label:', min_target)
            ConfusionMatrixDisplay(cm_p, display_labels=self.users_manager.test_samples.class_to_idx.keys()).plot()
            plt.title('Confusion matrix after pruning {} filters on poised data'.format(i))
            plt.savefig(os.path.join(path_exists(self.results_path), 'cm_poised_after_pruning_round().png'.format(self.rounds)))
            plt.close()

            ConfusionMatrixDisplay(cm_c, display_labels=self.users_manager.test_samples.class_to_idx.keys()).plot()
            plt.title('Confusion matrix after pruning {} filters on clean data'.format(i))
            plt.savefig(os.path.join(path_exists(self.results_path), 'cm_clean_after_pruning_round().png'.format(self.rounds)))
            plt.close()

            #for l in loss_dict.keys(): plt.plot(loss_dict[l], label=str(l))
            plt.plot(loss_val)
            plt.xlabel('Filters pruned, chosen {}'.format(best_index))
            plt.ylabel('Loss of estimated target label {}'.format(min_target))
            plt.axvline(best_index, linestyle='dashed', c='orange', label='Pruning limit')
            plt.axvline(np.argmax(abs_diff), linestyle='dashed', c='red', label='Max absolute clean/poised accuracy difference')
            plt.axvline(np.argmax(loss_val), linestyle='dashed', c='green', label='Min estimated loss')
            plt.legend()
            plt.savefig(os.path.join(path_exists(self.results_path), 'viz_pruning_trigger_loss_round{}.png'.format(self.rounds)), format='png')
            plt.close()
            # save_as_txt(list(loss_dict.values()), 'pruning_trigger_loss_per_target', os.path.join(path_exists(self.results_path)))
            save_as_txt(loss_val, 'pruning_trigger_loss_target', os.path.join(path_exists(self.results_path)))

            plt.plot(bd_success)
            plt.xlabel('Filters pruned, chosen {}'.format(best_index))
            plt.ylabel('Estimated backdoor accuracy')
            plt.axvline(best_index, linestyle='dashed', c='orange', label='Pruning limit')
            plt.axvline(np.argmax(abs_diff), linestyle='dashed', c='red', label='Max absolute clean/poised accuracy difference')
            plt.legend()
            plt.savefig(os.path.join(path_exists(self.results_path), 'viz_pruning_estimated_bd_success_round{}.png'.format(self.rounds)), format='png')
            plt.close()
            save_as_txt(bd_success, 'pruning_estimated_bd_success', os.path.join(path_exists(self.results_path)))

            print('Round', self.rounds, 'best index', best_index)
            # Load first model
            if 'soft' not in type_:
                prune.remove(getattr(net, layer), 'weight')
                net.load_state_dict(w)
                mask_ = torch.ones(net.state_dict()[layer + '.weight'].shape).cuda()
                mask_[f[:best_index+1]] = 0
                prune.custom_from_mask(getattr(net, layer), name='weight', mask=mask_)
            else:
                net.load_state_dict(w)
                mask_ = torch.ones(net.state_dict()[layer + '.weight'].shape).cuda()
                mask_[f[:best_index+1]] = 0.5
                with torch.no_grad():
                    net.state_dict()[layer+'.weight'] *= mask_

        plt.plot(pruning_acc_clean, label='Clean accuracy')
        plt.plot(pruning_acc_poise, label='Backdoor success')
        plt.xlabel('Number of pruned filters')
        plt.legend()
        plt.savefig(os.path.join(path_exists(self.results_path), 'viz_pruning_{}_{}.png'.format(name, type_)), format='png')
        plt.close()

        if 'reverse' in type_ and 'soft' not in type_:
            # Reset weights
            if '.' in layer:
                prune.remove(getattr(getattr(net, layer[0])[int(layer[1])], layer[2]), 'weight')
            else:
                prune.remove(getattr(net, layer), 'weight')
        # Save final model
        save_state_dict(net, 'Server_Model_pruned_{}'.format(type(net).__name__), path_exists(self.results_path))

        # Load first model
        # net.load_state_dict(w)
        """

    def server_pruning(self, layer, type_='L1', projection=False, plot=False, **kwargs):  # one layer for now
        self.results.update({'pruning_layer': layer, 'pruning_type': type_})
        self.results.update(kwargs)
        return self.pruning(self.server.net, layer, type_, name='Server_round{}'.format(self.rounds), projection=projection, plot=plot, **kwargs)

    def build_protective_mask(self, net, plot=False, **kwargs):
        self.results.update(kwargs)
        # Freeze weights
        for param in net.parameters():
            param.requires_grad = False

        shape = [kwargs['samples_mask']] + list(self.users_manager.test_samples[0][0].shape)
        common_trigger = []

        loss_dict = {int(k): np.zeros(kwargs['E_mask']) for k in self.users_manager.test_samples.class_to_idx.values()}
        criterion = torch.nn.CrossEntropyLoss()
        min_value = float('inf')
        index = int(np.round(self.users_manager.d * self.users_manager.d * kwargs['alpha']))
        for target in loss_dict.keys():
            train_samples_target = UniformSamePixelsData(shape, target)
            train_samples_target.data.requires_grad = True
            optimizer = torch.optim.SGD([train_samples_target.data], lr=kwargs['lr_mask'])
            dataloader = DataLoader(train_samples_target, batch_size=kwargs['B_mask'], shuffle=False)
            for epoch in range(kwargs['E_mask']):
                for d, t in dataloader:
                    optimizer.zero_grad()
                    if self.cuda:
                        d, t = d.cuda(), t.cuda()
                    output = net(d)
                    loss = criterion(output, t)
                    loss_dict[target][epoch] += loss.item()
                    loss.backward()
                    optimizer.step()

                loss_dict[target][epoch] /= len(dataloader)

            if loss_dict[target][-1] < min_value:
                min_target, min_value = target, loss_dict[target][-1]

            # Centralize and select
            mu = train_samples_target.data.mean(dim=[1, 2, 3], keepdim=True)
            x = torch.abs(train_samples_target.data - mu).mean(dim=[0])
            for c in range(self.users_manager.c):
                min_ = x[c].reshape(self.users_manager.d * self.users_manager.d).sort(descending=True).values[index]
                x[c] = torch.where(x >= min_, x[c], torch.Tensor([0]).float())
            common_trigger += [x]

        common_trigger = torch.stack(common_trigger)

        t_weights = 1/(2*(self.users_manager.o-1))*torch.ones(self.users_manager.o).reshape(-1, 1, 1, 1)
        t_weights[min_target] = torch.Tensor([1/2])
        common_trigger = torch.sum(t_weights*common_trigger, axis=0)

        for c in range(self.users_manager.c):
            # Center
            m = common_trigger[c].mean()
            min_ = (common_trigger[c]-m).abs().reshape(self.users_manager.d * self.users_manager.d).sort(descending=True).values[index]  # shape n
            common_trigger[c] = torch.where((common_trigger[c] - m).abs() >= min_,
                                            common_trigger[c],
                                            torch.Tensor([0]).float())
        

        # Test on clean and poised
        acc_c, cm = confusion_matrix_no_grad_mask_expand3(self.users_manager.test_samples, net, common_trigger,
                                                  samp=self.data_sampler('clean'), plot=True,
                                                  plot_file=os.path.join(self.results_path, 'cm_clean_wmask'), cuda=self.cuda,
                                                  plot_ex=os.path.join(self.results_path, 'clean_mask_ex.png'),
                                                  step=kwargs['step'])
        acc_p, cm = confusion_matrix_no_grad_mask_expand3(self.users_manager.test_samples, net, common_trigger,
                                                  samp=self.data_sampler('poised'), plot=True,
                                                  plot_file=os.path.join(self.results_path, 'cm_poised_wmask'), cuda=self.cuda,
                                                  plot_ex=os.path.join(self.results_path, 'poised_mask_ex.png'),
                                                  step=kwargs['step'])
        acc_bd = np.sum(cm[:, self.attack_target])/np.sum(cm)
        corr = 0
        if min_target == self.attack_target:
            corr = 1

        if plot:
            if len(common_trigger.shape) > 2:
                common_trigger = torch.permute(common_trigger, (1, 2, 0))
                color = 'viridis'
            else:
                color = 'gray'
            plt.imshow(common_trigger.squeeze().detach(), interpolation=None, cmap=color)
            plt.colorbar(shrink=0.90)
            plt.title('New clean acc {} and poised acc {}'.format(acc_c, acc_p))

            n = torch.where(self.users_manager.pattern==255)
            for x,y in zip(n[0], n[1]):
                plt.scatter(y,x, color='r', marker='+')

            plt.savefig(os.path.join(path_exists(self.results_path), 'common_reverse_trigger.png'), format='png',
                        bbox_inches='tight')
            plt.close()

        return acc_c, acc_p, acc_bd, corr
