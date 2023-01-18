import os
import sys
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import torch
import random
import numpy as np

from actors.user import User
from custom_datasets.custom_datasets import CIFAR10_custom, GTSRB_custom, FEMNIST_custom


class UsersManager(Dataset):

    def __init__(self, dataset, nb_users, path=os.getcwd()):
        self.dataset = dataset
        self.clean_train_indexes = []
        self.clean_test_indexes = []

        self.poised_train_indexes = []
        self.poised_test_indexes = []

        if self.dataset == 'MNIST':
            self.train_samples = EMNIST(path, split='digits', download=True, transform=ToTensor())
            self.test_samples = EMNIST(path, split='digits', train=False, download=True, transform=ToTensor())
            self.c, self.d, self.o = 1, 28, 10
        elif self.dataset == 'CIFAR10':
            self.train_samples = CIFAR10_custom(path, download=True,
                                                transform=Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
            self.test_samples = CIFAR10_custom(path, train=False, download=True,
                                        transform=Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
            self.c, self.d, self.o = 3, 32, 10
        elif self.dataset == 'FashionMNIST':
            self.train_samples = FashionMNIST(path, download=True, transform=ToTensor())
            self.test_samples = FashionMNIST(path, train=False, download=True, transform=ToTensor())
            self.c, self.d, self.o = 1, 28, 10
        elif self.dataset == 'GTSRB':
            self.train_samples = GTSRB_custom(path, split='train', download=True,
                                              #transform=Normalize((0.3420, 0.3129, 0.3213), (0.2740, 0.2611, 0.2662)))
                                              transform=None)
            self.test_samples = GTSRB_custom(path, split='test', download=True,
                                             #transform=Normalize((0.3420, 0.3129, 0.3213), (0.2740, 0.2611, 0.2662)))
                                             transform=None)
            self.c, self.d, self.o = 3, 32, len(self.train_samples.class_to_idx)
        elif self.dataset == 'FEMNIST':
            self.train_samples = FEMNIST_custom(path, download=False, transform=None)
            self.test_samples = FEMNIST_custom(path, train=False, download=False, transform=None)
            self.c, self.d, self.o = 1, 28, len(self.train_samples.class_to_idx)
        else:
            sys.exit('Choose available dataset to load.')

        self.nb_users = nb_users
        self.poised_users = None
        self.users_accuracy_clean = None
        self.users_accuracy_poise = None
        self.users = []
        self.users_niid_groups = None
        self.pattern = None

    def __len__(self):
        return len(self.users)

    def create_users(self):
        if self.dataset == 'FEMNIST':
            # Datset is not balanced between users: max 109 and 21 samples for train and test respectively.
            # Take users with most data
            for idx in np.argsort(np.bincount(self.train_samples.writers))[::-1][:self.nb_users]:
                user_train_indexes = np.where(self.train_samples.writers == idx)[0].tolist()
                user_test_indexes = np.where(self.test_samples.writers == idx)[0].tolist()
                self.clean_train_indexes += user_train_indexes
                self.clean_test_indexes += user_test_indexes
                self.users += [User(idx, user_train_indexes, user_test_indexes)]
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
            self.users += [User(idx, user_train_indexes, user_test_indexes)]
        return self.users

    def create_users_label_skew(self, lb_per_user):
        nb_groups = int(len(self.train_samples.class_to_idx)/lb_per_user) # number of groups with same label skew
        len_group = int(self.nb_users/nb_groups) # number of clients in each group
        self.users_niid_groups = {k: list(range(k*len_group, (k+1)*len_group)) for k in range(nb_groups)} # keys: groups, values: users ids
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
                self.users += [User(id_, user_train_indexes, user_test_indexes)]

        return None

    def poise_users(self, target, nb_poised_users, p_poise, pattern='single_pixel', no_poise_class=[]):
        # Poise first users.
        #self.poised_users = np.arange(0, nb_poised_users, 1)
        # No, now poison every group of non-IID.
        if not self.users_niid_groups:
            random.seed(0)
            self.poised_users = random.sample(range(self.nb_users), k=nb_poised_users)
        else:
            self.poised_users = [self.users_niid_groups[k][:nb_poised_users // len(self.users_niid_groups)] for k in
                                 self.users_niid_groups.keys()]
            # flatten
            self.poised_users = [item for sublist in self.poised_users for item in sublist]
            # add the rest of poised users from first group
            if nb_poised_users % len(self.users_niid_groups) != 0:
                self.poised_users += self.users_niid_groups[0][-int(nb_poised_users % len(self.users_niid_groups)):]

        for p in self.poised_users:
            self.users[p].attacked = True
            for c in self.train_samples.class_to_idx.values():
                if c not in no_poise_class: # won't be poisoning some labels... not the same number of poised images than usual but, important?
                    target_indexes = list(set(torch.where(self.train_samples.targets == int(c))[0].tolist())
                                       .intersection(set(self.users[p].train_indexes)))
                    self.poised_train_indexes += random.sample(target_indexes, k=round(p_poise*len(target_indexes)))
            #self.poised_train_indexes += random.sample(self.users[p].train_indexes,
            #                                               k=round(p_poise * len(self.users[p].train_indexes)))

            #self.poised_test_indexes += random.sample(self.users[p].test_indexes, k=round(p_poise*len(self.users[p].test_indexes)))
        # We will divide test set in 2: poised and clean images
        self.poised_test_indexes += random.sample(self.clean_test_indexes, k=round(0.5*len(self.clean_test_indexes)))

        # Set the clean indexes
        self.clean_train_indexes = list(set(self.clean_train_indexes)-set(self.poised_train_indexes))
        self.clean_test_indexes = list(set(self.clean_test_indexes)-set(self.poised_test_indexes))

        if len(self.test_samples.data.shape[1:]) < 3: #MNIST, FashionMNIST
            l, L = self.test_samples.data.shape[1:]
            c = 1
        else: # Color images
            l, L, c = self.test_samples.data.shape[1:]

        mask = torch.zeros(l, L, 1)
        if pattern == 'single_pixel':
            pat = torch.Tensor([[1]])
            mask[-pat.shape[0]-1:-1, -pat.shape[1]-1:-1, 0] = pat
            self.pattern = 255*mask

        elif pattern == 'yellow_sticker':
            pat = torch.ones((2, 2))
            mask[15:15+pat.shape[0], 15:15+pat.shape[1], 0] = pat
            self.pattern = torch.zeros(l, L, c)
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 0] = 255 * pat
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 1] = 255 * pat

        elif pattern == 'yellow_sticker_plus':
            pat = torch.ones((3, 3))
            mask[15:15+pat.shape[0], 15:15+pat.shape[1], 0] = pat
            self.pattern = torch.zeros(l, L, c)
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 0] = 255 * pat
            self.pattern[15:15 + pat.shape[0], 15:15 + pat.shape[1], 1] = 255 * pat

        elif pattern == 'emoji':
            pat = torch.Tensor([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1],
                                [0, 1, 1, 0, 1, 0, 1, 1, 0],
                                [0, 0, 1, 0, 1, 0, 1, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 0, 1, 0, 1, 0, 1, 1]])
            mask[-pat.shape[0] - 1:-1, -pat.shape[1] - 1:-1, 0] = pat
            self.pattern = 255 * mask

        elif pattern == 'yellow_emoji':
            pat = torch.Tensor([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [1, 0, 1, 1, 1, 1, 1, 0, 1],
                                [0, 1, 1, 0, 1, 0, 1, 1, 0],
                                [0, 0, 1, 0, 1, 0, 1, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 0, 1, 0, 1, 0, 1, 1]])
            mask[-pat.shape[0]-1:-1, -pat.shape[1]-1:-1, 0] = pat
            self.pattern = torch.zeros(l, L, c)
            self.pattern[-pat.shape[0] - 1:-1, -pat.shape[1] - 1:-1, 0] = 255 * pat
            self.pattern[-pat.shape[0] - 1:-1, -pat.shape[1] - 1:-1, 1] = 255 * pat

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

        elif pattern == 'bad_pattern':
            # BadNets pattern
            pat = torch.Tensor([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 1]])
            mask[-pat.shape[0] - 1:-1, -pat.shape[1] - 1:-1, 0] = pat
            self.pattern = 255 * mask

        elif pattern == 'bad_pattern_plus':
            # BadNets pattern
            pat = torch.Tensor([[0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1]])
            mask[-pat.shape[0] - 1:-1, -pat.shape[1] - 1:-1, 0] = pat
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

        elif pattern == 'half_emoji': #not updated
            # Add half invader pixel pattern to a copy of all of the test images
            pattern_mask = torch.Tensor([[0, 1, 0, 0, 0],
                                         [1, 1, 1, 0, 1],
                                         [1, 0, 1, 1, 0],
                                         [1, 0, 1, 0, 0],
                                         [1, 1, 1, 1, 0],
                                         [0, 1, 0, 1, 1]])

        else:
            exit()

        if self.dataset == 'MNIST':  # MNIST
            self.pattern = torch.permute(self.pattern, (1, 0, 2)) # MNIST images are rotated
            mask = torch.permute(mask, (1, 0, 2))
        if c == 1:
            self.pattern = self.pattern.squeeze()
            mask = mask.squeeze()

        type_ = self.train_samples.data[0].dtype
        type_t = self.train_samples.targets[0].dtype

        self.train_samples.data[self.poised_train_indexes] = \
            torch.where(mask != 0, self.pattern.type(type_), self.train_samples.data[self.poised_train_indexes])

        self.test_samples.data[self.poised_test_indexes] = \
            torch.where(mask != 0, self.pattern.type(type_), self.test_samples.data[self.poised_test_indexes])

        self.train_samples.targets[self.poised_train_indexes] = target * torch.ones(len(self.poised_train_indexes),
                                                                                    dtype=type_t)

        #self.test_samples.targets[self.poised_test_indexes] = target * torch.ones(len(self.poised_test_indexes),
        #                                                                          dtype=torch.int64)
        # Last line undone because CM matrix
        return None
