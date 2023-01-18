from tools.save_tools import add_to_dict

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy import cov, dot, argsort
from numpy.linalg import svd


class Server:
    def __init__(self, net):
        self.net = net
        self.average_weights = {}
        self.received_weights = {}
        self.sorting_indexes = {}
        self.clean_accuracy_per_round = []
        self.bd_success_per_round = []
        self.clean_conf_per_round = []
        self.bd_conf_per_round = []
        self.activations = {}

    def receive_from_user(self, params, nk, aggregation='avg', **kwargs):
        if aggregation == 'dp':
            assert 'S' in kwargs, 'Define S if using DP. It should be non negative.'
            assert 'sigma' in kwargs, 'Define sigma if using DP. It should be non negative.'
            assert 'K' in kwargs, 'Send K to function.'
            norm = torch.sqrt(torch.sum(torch.Tensor([torch.norm(torch.flatten(params[layer]), p=2)**2 for layer in params])))
        for layer in params:
            if aggregation == 'median' or aggregation == 'krum':
                add_to_dict(params[layer], layer, self.received_weights)
            elif aggregation == 'avg':
                add_to_dict(nk * params[layer], layer, self.received_weights)
            elif aggregation == 'dp':
                # Add clipping and noise
                add_to_dict(min(1, kwargs['S']/norm) * params[layer] +
                            torch.normal(mean=0,
                                         std=kwargs['sigma'] * 2 * kwargs['S'] / np.sqrt(kwargs['K']),
                                         size=params[layer].shape).cuda(),
                            layer, self.received_weights)
        return None

    def check_weigths(self):
        # torch.cov Estimates the covariance matrix of the variables given by the input matrix, where rows are the
        # variables and columns are the observations.
        for layer in self.received_weights:
            weights = torch.vstack(self.received_weights[layer])
            cov = torch.cov(weights) #normalize
            u,s,v = np.linalg.svd(cov)
            sort = torch.argsort(torch.dot(weights, v[0, :])) #an anomalous instance that deviates from the correlation structure will have a large value
            # of the projection of each data instance along the principal components with low variance
            self.sorting_indexes[layer] = sort

    def update_average_weights(self, N, update=True, aggregation='avg', krum_f=None):
        # Average parameters
        if aggregation == 'avg' or aggregation == 'dp':
            for layer in self.received_weights:
                #self.average_weights[layer] = self.net.state_dict()[layer]\
                #                          - (1/N)*torch.sum(torch.stack(self.received_weights[layer]), 0)

                self.average_weights[layer] = (1/N)*torch.sum(torch.stack(self.received_weights[layer]), 0)

        elif aggregation == 'krum':
            # Decided to compute scores considering all weight matrices
            scores = []
            weights_per_user = []
            with torch.no_grad():
                for layer in self.received_weights.keys():
                    x = [torch.flatten(w) for w in self.received_weights[layer]]
                    weights_per_user += [torch.vstack(x)]
                weights_per_user = torch.hstack(weights_per_user)
                # Compute squared L2 norm (Frobenius norm for matrices)
                X = weights_per_user.detach().cpu().numpy()
                D2 = squareform(pdist(X, metric=lambda u, v: ((u - v) ** 2).sum()))

            for i in range(D2.shape[0]):
                dist_i = D2[i, :]
                # ascending sort
                dist_i.sort()
                # K-f-2 smallest distances : -1 because slicing is not inclusive
                score_i = dist_i[:D2.shape[0] - krum_f - 1].sum()
                scores += [score_i]
            i_star = np.argmin(scores)
            for layer in self.received_weights.keys():
                self.average_weights[layer] = self.net.state_dict()[layer] - self.received_weights[layer][i_star]

        elif aggregation == 'median':
            for layer in self.received_weights.keys():
                medians = torch.median(torch.vstack([p.unsqueeze(0) for p in self.received_weights[layer]]), dim=0)[0]
                self.average_weights[layer] = self.net.state_dict()[layer] - medians

        if update:
            # Update server net
            self.net.load_state_dict(self.average_weights)
            # Reset received weights to avoid accumulation between rounds
            self.received_weights = {}
            return None
        else:
            # Just return average, without updating
            return self.average_weights

    """
    def server_controled_pruning(self, N, layer):  # one layer for now
        w = self.net.state_dict()
        pruning_acc_clean_before = []
        pruning_acc_poise_before = []

        pruning_acc_clean_after = []
        pruning_acc_poise_after = []

        #avg = self.update_average_weights(N, update=False)
        weights = self.net.state_dict()[layer + '.weight'].detach()

        weights = weights.T - weights.mean(axis=1)
        C = cov(weights, bias=True)  # rows: vars, column: observation
        u, s, v = svd(C, full_matrices=True)

        # Je dois avoir 128 valeurs a trier parce que c'est la dim de mes observations
        f = argsort(dot(weights.T, v[0, :]))[::-1]  # descending

        malicious = []
        for i_malicious in range(len(f)):
            malicious += [i_malicious]
            mu_m = torch.mean([self.received_weights[layer][i] for i in malicious])
            delta = [i_malicious]-self.received_weights[layer][]
        mask_ = torch.ones(self.server.net.state_dict()[layer + '.weight'].shape).cuda()
        for j in f:
            mask_[j] = 0
            prune.custom_from_mask(getattr(self.server.net, layer), name='weight', mask=mask_)
            pruning_acc_clean += [self.test_server('clean')]
            pruning_acc_poise += [self.test_server('poised')]

        save_as_txt(pruning_acc_clean, 'pruning_acc_clean_{}'.format(type_), self.results_path)
        save_as_txt(pruning_acc_poise, 'pruning_acc_poise_{}'.format(type_), self.results_path)

        # Reset weights
        prune.remove(getattr(self.server.net, layer), 'weight')
        self.server.net.load_state_dict(w)

        # Plot
        plt.plot(pruning_acc_clean, label='Clean accuracy')
        plt.plot(pruning_acc_poise, label='Backdoor success')
        plt.xlabel('Number of pruned filters')
        plt.legend()
        plt.savefig(os.path.join(self.results_path, 'viz_pruning_{}.png'.format(type_)), format='png')
        plt.close()

    """
