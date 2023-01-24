import random
import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, sampler
import numpy as np

from metrics import confusion_matrix_no_grad, confusion_matrix_no_grad_protected
from actors import UsersManager
from actors import Server


class BackdooredFedAVG:

    def __init__(self, net, nb_users, attack_target, attack_type, nb_attack, p_attack, dataset, data_path=os.getcwd(),
                 label_skew=False, server_preload_net=None, cuda=False):
        """
        Initializes class BackdoorFedAVG.
        :param net: (torch.nn.Module) Network model class to be used by clients.
        :param nb_users: (int) Number of clients to simulate.
        :param attack_target: (int) Backdoor attack target.
        :param attack_type: (str) Backdoor attack pattern. Supports 'square', 'cross', 'copyright' and 'yellow_sticker'.
        :param nb_attack: (int) Number of malicious clients to be simulated.
        :param p_attack: (float) Proportion of poised local dataset for each malicious client.
        :param dataset: (str) Dataset to be used. Supported options are 'MNIST', 'FashionMNIST', 'FEMNIST' and 'GTSBR'.
        :param data_path: (str) Path where dataset is stored or will be downloaded. For dataset 'FEMNIST', file
        'emnist-digits.mat' must be located in file."
        :param label_skew: (int) If label skewed distribution (non-IID), number of labels per client. Default False.
        :param server_preload_net: (str) (Optional) Path to .pth file if global model already trained.
        :param cuda: (bool) True to use.
        """
        self.device = torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')

        self.users_manager = UsersManager(dataset, nb_users, data_path)
        self.net = net(in_chan=self.users_manager.channels, in_d=self.users_manager.dim, out_d=self.users_manager.out).to(self.device)
        self.server = Server(net(in_chan=self.users_manager.channels, in_d=self.users_manager.dim,
                                 out_d=self.users_manager.out).to(self.device))
        self.nb_users = nb_users
        if server_preload_net:
            print('Preloaded server net')
            self.server.net.load_state_dict(torch.load(server_preload_net))
            self.server.net.to(self.device)

        self.attack_target = attack_target
        self.attack_type = attack_type
        self.nb_attack = nb_attack
        self.p_attack = p_attack

        self.label_skew = label_skew
        if self.label_skew:
            self.users_manager.create_users_label_skew(self.label_skew)
        else:
            self.users_manager.create_users()

        self.users_manager.poise_users(self.attack_target, self.nb_attack, self.p_attack, pattern=self.attack_type)

        self.rounds = 0
        if not os.path.exists(os.path.join(os.getcwd(), "results")):
            os.mkdir(os.path.join(os.getcwd(), "results"))
        self.results_path = os.path.join(os.getcwd(), "results")

    def user_update(self, user, E, B, lr):
        user_net = self.net
        user_net.load_state_dict(self.server.net.state_dict())
        user_net.to(self.device)
        samples = sampler.SubsetRandomSampler(user.train_indexes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(user_net.parameters(), lr=lr)

        for epoch in range(E):
            dataloader_train = DataLoader(self.users_manager.train_samples, batch_size=B, sampler=samples,
                                          shuffle=False, num_workers=0)
            for d, t in dataloader_train:
                d, t = d.to(self.device), t.to(self.device)
                optimizer.zero_grad()
                output = user_net(d)
                loss = criterion(output, t)
                loss.backward()
                optimizer.step()

        return user_net.state_dict()

    def backdoored_fedvag(self, rounds, E, B, C=0.1, lr=0.01, save_net=False):

        best_round, best_acc = 0, 0

        for round_ in range(1, rounds + 1):
            self.rounds = round_
            N = 0
            clean_acc, _ = confusion_matrix_no_grad(self.users_manager.test_samples, self.server.net,
                                                 samp=self.data_sampler('clean'), device=self.device)
            if self.users_manager.poised_train_indexes:
                poised_acc, cm = confusion_matrix_no_grad(self.users_manager.test_samples, self.server.net,
                                                 samp=self.data_sampler('poised'), device=self.device)
                bd_success = np.sum(cm[:, self.attack_target]) / np.sum(cm)

            # Remember round of best accuracy : will be associated to accuracy before training
            if clean_acc > best_acc:
                best_acc, best_round = clean_acc, round_
                if self.users_manager.poised_train_indexes:
                    best_poised, best_bd = poised_acc, bd_success
                if save_net:
                    torch.save(self.server.net.state_dict(), os.getcwd())

            sampled = self.select_users(C=C)
            for user in sampled:
                params = self.user_update(user, E, B, lr)
                N += user.nk
                # Send to server
                self.server.receive_from_user(params, user.nk)

            self.server.update_average_weights(N=N)

        # Rename best net
        if save_net and os.path.exists(os.path.join(self.results_path, 'server_net.pth')):
            # Keep best net
            self.server.net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'server_net.pth')))
            os.rename(os.path.join(os.getcwd(), 'server_net.pth'),
                      os.path.join(os.getcwd(), 'Server_net_best_round{}.pth'.format(best_round)))
        if self.users_manager.poised_train_indexes:
            return best_acc, best_poised, best_bd, best_round
        else:
            return best_acc, best_round

    def select_users(self, C=0.1):
        total_nb_users = len(self.users_manager)
        K = max(1, round(C * total_nb_users))
        return random.sample(self.users_manager.users, k=K)

    def data_sampler(self, type_='clean'):
        if type_ == 'clean':
            return sampler.SubsetRandomSampler(self.users_manager.clean_test_indexes)
        elif type_ == 'poised':
            return sampler.SubsetRandomSampler(self.users_manager.poised_test_indexes)
        elif isinstance(type_, list):
            return sampler.SubsetRandomSampler(type_)
        else:
            sys.exit('Chose which data to sample.')

    def test_defense(self, samples, iterations, lr_bandage, gamma, save=False, **kwargs):

        common_trigger = self.server.build_protective_bandage(samples, iterations, lr_bandage, gamma,
                                                              channels=self.users_manager.channels,
                                                              dim=self.users_manager.dim,
                                                              targets=self.users_manager.out,
                                                              device=self.device)

        # Test on clean and poised
        acc_c, cm = confusion_matrix_no_grad_protected(self.users_manager.test_samples, self.server.net, common_trigger,
                                                       samp=self.data_sampler('clean'), device=self.device,
                                                       step=kwargs.get('kernel_size', 1))
        acc_p, cm = confusion_matrix_no_grad_protected(self.users_manager.test_samples, self.server.net, common_trigger,
                                                       samp=self.data_sampler('poised'), device=self.device,
                                                       step=kwargs.get('kernel_size', 1))
        acc_bd = np.sum(cm[:, self.attack_target]) / np.sum(cm)

        if save:
            if len(common_trigger.shape) > 2:
                common_trigger = torch.permute(common_trigger, (1, 2, 0))
                color = 'viridis'
            else:
                color = 'gray'
            plt.imshow(common_trigger.squeeze().detach(), interpolation=None, cmap=color)

            n = torch.where(self.users_manager.pattern == 255)
            for x, y in zip(n[0], n[1]):
                plt.scatter(y, x, color='r', marker='+', label='Hidden trigger')
            plt.legend()
            plt.colorbar(shrink=0.90)
            plt.title('Estimated hidden trigger')
            plt.savefig(os.getcwd(), 'common_reverse_trigger.png', format='png', bbox_inches='tight')
            plt.close()
        return acc_c, acc_bd, acc_p

