import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import EMNIST, FashionMNIST
from torchvision.transforms import ToTensor
import torch
import random
import numpy as np

from custom_datasets import GTSRBcustom, FEMNISTcustom, UniformSamePixelsData
from metrics import confusion_matrix_no_grad_protected


class Server:
    """
    Class that represent the central server in federated learning.
    """
    def __init__(self, net):
        """
        Initializes class Server.
        :param net: class torch.nn.Module
        Network to be used by clients.
        """
        self.net = net
        self.average_weights = {}
        self.received_weights = {k: [] for k in net.state_dict().keys()}
        self.activations = {}

    def receive_from_user(self, params, nk):
        """
        Server accumulates parameters received by a user.
        :param params: (dict) Dictionary whose keys are the client's model layers names and values are their
        corresponding parameters.
        :param nk: (int) Number of samples of client.
        :return: None
        """
        for layer in params:
            self.received_weights[layer] += [nk * params[layer]]
        return None

    def update_average_weights(self, N):
        """
        Server averages the received parameters.
        :param N: (int) Total number of samples received.
        :return: None
        """
        for layer in self.received_weights:
            self.average_weights[layer] = (1/N)*torch.sum(torch.stack(self.received_weights[layer]), 0)

        # Update the server net
        self.net.load_state_dict(self.average_weights)
        # Reset received weights to avoid accumulation between rounds
        self.received_weights = {}
        return None

    def build_protective_bandage(self, samples, iterations, lr_bandage, gamma, channels, dim, targets, device='cpu'):
        """
        Server computes hidden trigger estimation.
        :param samples: (int) Number of samples of random uniform data to generate.
        :param iterations: (int) Number of iterations for estimation.
        :param lr_bandage: (int) Learning ratefor estimation.
        :param gamma: (float) Percentage of information to keep of estimation.
        :param channels: (int) Number of channels of data. 1 for MNIST, FashionMNIST and FEMNIST. 3 for GTSRB.
        :param dim: (int) Dimension of original data. 28x28 for MNIST, FashionMNIST and FEMNIST. 32x32 for GTSRB.
        :param targets: (int) Number of targets of original task.
        :param device: (str) Compute on 'gpu' ot 'cpu'.
        :return: Hidden trigger estimation.
        """
        # Freeze weights
        for param in self.net.parameters():
            param.requires_grad = False

        shape = [samples, channels, dim, dim]

        common_trigger = []

        loss_dict = {int(k): np.zeros(iterations) for k in range(targets)}
        criterion = torch.nn.CrossEntropyLoss()
        min_value = float('inf')
        for target in loss_dict.keys():
            train_samples_target = UniformSamePixelsData(shape, target)
            train_samples_target.data.requires_grad = True
            optimizer = torch.optim.SGD([train_samples_target.data], lr=lr_bandage)
            dataloader = DataLoader(train_samples_target, batch_size=1, shuffle=False)
            for epoch in range(iterations):
                for d, t in dataloader:
                    optimizer.zero_grad()
                    d, t = d.to(device), t.to(device)
                    output = self.net(d)
                    loss = criterion(output, t)
                    loss_dict[target][epoch] += loss.item()
                    loss.backward()
                    optimizer.step()

                loss_dict[target][epoch] /= len(dataloader)

            if loss_dict[target][-1] < min_value:
                min_target, min_value = target, loss_dict[target][-1]

            mu = train_samples_target.data.mean(dim=[1, 2, 3], keepdim=True)
            x = torch.abs(train_samples_target.data - mu).mean(dim=0)

            common_trigger += [x]

        common_trigger = torch.stack(common_trigger)
        t_weights = 1 / (2 * (targets - 1)) * torch.ones(targets).reshape(-1, 1, 1, 1)
        t_weights[min_target] = torch.Tensor([1 / 2])
        common_trigger = torch.sum(t_weights * common_trigger, axis=0)
        index = int(np.round(dim * dim * gamma))
        for i in range(channels):
            min_ = common_trigger[i, :, :].reshape(dim * dim).sort(descending=True).values[index]  # shape n
            common_trigger[i] = torch.where(common_trigger[i] >= min_, common_trigger[i], torch.Tensor([0]).float())

        return common_trigger


class User:
    """
    Class to represent a client in the system.

    Attributes:
        idx (int): Index of the client in the system.
        train_indexes (List[int]): Indexes of the samples used for training.
        attacked (bool): Flag to indicate whether the client has been attacked.
        nk (int): Number of training samples.
    """
    def __init__(self, idx, train_indexes):
        self.idx = idx
        self.train_indexes = train_indexes
        self.attacked = False
        self.nk = len(train_indexes)


class UsersManager(Dataset):
    """
    Class to manage clients' data.

    Attributes:
        dataset (string): The name of the dataset to be loaded. The class supports 'MNIST', 'FashionMNIST', 'GTSRB'
        and 'FEMNIST'.
        nb_users (int): The number of users to be created.
        attacked (bool): Flag to indicate whether the client has been attacked.
        path (string): The path to the directory where the data should be saved.
    """
    def __init__(self, dataset, nb_users, path=os.getcwd()):
        self.dataset = dataset
        self.clean_train_indexes = []
        self.clean_test_indexes = []

        self.poised_train_indexes = []
        self.poised_test_indexes = []
        self.channels = None
        self.dim =None
        self.out = None

        if self.dataset == 'MNIST':
            self.train_samples = EMNIST(path, split='digits', download=True, transform=ToTensor())
            self.test_samples = EMNIST(path, split='digits', train=False, download=True, transform=ToTensor())
            self.channels, self.dim, self.out = 1, 28, 10
        elif self.dataset == 'FashionMNIST':
            self.train_samples = FashionMNIST(path, download=True, transform=ToTensor())
            self.test_samples = FashionMNIST(path, train=False, download=True, transform=ToTensor())
            self.channels, self.dim, self.out = 1, 28, 10
        elif self.dataset == 'GTSRB':
            self.train_samples = GTSRBcustom(path, split='train', download=True,
                                              transform=None)
            self.test_samples = GTSRBcustom(path, split='test', download=True,
                                             transform=None)
            self.channels, self.dim, self.out = 3, 32, len(self.train_samples.class_to_idx)
        elif self.dataset == 'FEMNIST':
            self.train_samples = FEMNISTcustom(path, transform=None)
            self.test_samples = FEMNISTcustom(path, train=False, transform=None)
            self.channels, self.dim, self.out = 1, 28, len(self.train_samples.class_to_idx)
        else:
            sys.exit("Choose available dataset to load. Possible choices are 'MNIST', 'FashionMNIST', 'GTSRB' and"
                     " 'FEMNIST'. ")

        self.nb_users = nb_users
        self.poised_users = None
        self.users = []
        self.users_niid_groups = None
        self.pattern = None

    def __len__(self):
        return len(self.users)

    def create_users(self):
        """
        Creates the specified number of users and distributes the data among them.
        :return: (List[User]) List of clients.
        """
        if self.dataset == 'FEMNIST':
            # Datset is not balanced between users: max 109 and 21 samples for train and test respectively.
            # We take clients with most data.
            for idx in np.argsort(np.bincount(self.train_samples.writers))[::-1][:self.nb_users]:
                user_train_indexes = np.where(self.train_samples.writers == idx)[0].tolist()
                user_test_indexes = np.where(self.test_samples.writers == idx)[0].tolist()
                self.clean_train_indexes += user_train_indexes
                self.clean_test_indexes += user_test_indexes
                self.users += [User(idx, user_train_indexes)]
            return self.users

        for idx in range(self.nb_users):
            user_train_indexes = []
            user_test_indexes = []
            for c in self.train_samples.class_to_idx.values():
                indexes_train = torch.where(self.train_samples.targets == int(c))[0]
                indexes_test = torch.where(self.test_samples.targets == int(c))[0]

                # Each user will have nb_per_user samples per class
                nb_per_user_train = len(indexes_train) // self.nb_users
                nb_per_user_test = len(indexes_test) // self.nb_users

                user_train_indexes += indexes_train[int(idx*nb_per_user_train):int((idx+1)*nb_per_user_train)].tolist()
                user_test_indexes += indexes_test[int(idx*nb_per_user_test):int((idx+1)*nb_per_user_test)].tolist()

            self.clean_train_indexes += user_train_indexes
            self.clean_test_indexes += user_test_indexes
            self.users += [User(idx, user_train_indexes)]
        return self.users

    def create_users_label_skew(self, lb_per_user):
        """
        Creates a label skew distribution among clients.
        :param lb_per_user: Number of labels per client.
        :return: None
        """
        # Number of groups of clients with the same label skewed distribution.
        nb_groups = int(len(self.train_samples.class_to_idx)/lb_per_user)
        # Number of clients in each group.
        len_group = int(self.nb_users/nb_groups)
        # Keys: groups, values: Users ids
        self.users_niid_groups = {k: list(range(k*len_group, (k+1)*len_group)) for k in range(nb_groups)}
        labels = [list(self.train_samples.class_to_idx.values())[lb_per_user*i:lb_per_user*(i+1)] for i in range(nb_groups)]
        rest = len(self.train_samples.class_to_idx)%lb_per_user
        if rest != 0:
            labels[0] += list(self.train_samples.class_to_idx.values())[-rest:]

        for g in self.users_niid_groups:
            for i, id_ in enumerate(self.users_niid_groups[g]):
                user_train_indexes = []
                user_test_indexes = []
                for c in labels[g]:
                    indexes_train = torch.where(self.train_samples.targets == int(c))[0]
                    indexes_test = torch.where(self.test_samples.targets == int(c))[0]

                    # Each user will have nb_per_user samples per class
                    nb_per_user_train = len(indexes_train) // len_group
                    nb_per_user_test = len(indexes_test) // len_group

                    user_train_indexes += indexes_train[
                                          int(i * nb_per_user_train):int((i + 1) * nb_per_user_train)].tolist()
                    user_test_indexes += indexes_test[
                                         int(i * nb_per_user_test):int((i + 1) * nb_per_user_test)].tolist()

                self.clean_train_indexes += user_train_indexes
                self.clean_test_indexes += user_test_indexes
                self.users += [User(id_, user_train_indexes)]

        return None

    def poise_users(self, target, nb_poised_users, p_poise, pattern):
        """
        Poises clients' data with attack pattern.
        :param target: (int) Attack target class.
        :param nb_poised_users: (int) Number of clients to poise.
        :param p_poise: (float) Proportion of malicious clients' samples to poise.
        :param pattern: (string) Attack pattern. The method supports 'yellow_sticker', 'cross', 'square' and
        'copyright'. 'yellow_sticker' is only meant for color images (3 channels).
        :return: None
        """
        # Case for which data is IID.
        if not self.users_niid_groups:
            self.poised_users = random.sample(range(self.nb_users), k=nb_poised_users)
        # If data non-IID, choose malicious clients from each skewed distribution.
        else:
            self.poised_users = [self.users_niid_groups[k][:nb_poised_users // len(self.users_niid_groups)] for k in
                                 self.users_niid_groups.keys()]
            self.poised_users = [item for sublist in self.poised_users for item in sublist]
            # Add the rest of poised users from first group
            if nb_poised_users % len(self.users_niid_groups) != 0:
                self.poised_users += self.users_niid_groups[0][-int(nb_poised_users % len(self.users_niid_groups)):]

        for p in self.poised_users:
            self.users[p].attacked = True
            for c in self.train_samples.class_to_idx.values():
                target_indexes = list(set(torch.where(self.train_samples.targets == int(c))[0].tolist())
                                       .intersection(set(self.users[p].train_indexes)))
                self.poised_train_indexes += random.sample(target_indexes, k=round(p_poise*len(target_indexes)))

        # We will divide test set in 2: poised and clean images
        self.poised_test_indexes += random.sample(self.clean_test_indexes, k=round(0.5*len(self.clean_test_indexes)))

        # Remove poised indexes from the clean indexes.
        self.clean_train_indexes = list(set(self.clean_train_indexes)-set(self.poised_train_indexes))
        self.clean_test_indexes = list(set(self.clean_test_indexes)-set(self.poised_test_indexes))

        mask = torch.zeros(self.dim, self.dim, 1)
        if pattern == 'yellow_sticker':
            pat = torch.ones((2, 2))
            mask[15:15+pat.shape[0], 15:15+pat.shape[1], 0] = pat
            self.pattern = torch.zeros(self.dim, self.dim, self.channels)
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 0] = 255 * pat
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 1] = 255 * pat

        elif pattern == 'cross':
            pat = torch.Tensor([[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]])
            mask[1:1+pat.shape[0], 1:1+pat.shape[1], 0] = pat
            self.pattern = 255 * mask

        elif pattern == 'square':
            pat = torch.Tensor([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])
            mask[-pat.shape[0] - 2:-2, -pat.shape[1] - 2:-2, 0] = pat
            self.pattern = 255 * mask

        elif pattern == 'copyright':
            pat = torch.Tensor([[0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 1, 0, 0, 0, 0, 1, 1],
                                [1, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0, 1],
                                [0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 1, 1, 0, 0, 1, 0, 0]])
            mask[-pat.shape[0] - 2:-2, -pat.shape[1] - 2:-2, 0] = pat
            self.pattern = 255 * mask

        else:
            exit()
        # MNIST images are rotated
        if self.dataset == 'MNIST':
            self.pattern = torch.permute(self.pattern, (1, 0, 2))
            mask = torch.permute(mask, (1, 0, 2))
        if self.channels == 1:
            mask = mask.squeeze()

        type_ = self.train_samples.data[0].dtype
        type_t = self.train_samples.targets[0].dtype

        self.train_samples.data[self.poised_train_indexes] = \
            torch.where(mask != 0, self.pattern.squeeze().type(type_), self.train_samples.data[self.poised_train_indexes])

        self.test_samples.data[self.poised_test_indexes] = \
            torch.where(mask != 0, self.pattern.squeeze().type(type_), self.test_samples.data[self.poised_test_indexes])

        self.train_samples.targets[self.poised_train_indexes] = target * torch.ones(len(self.poised_train_indexes),
                                                                                    dtype=type_t)

        return None
