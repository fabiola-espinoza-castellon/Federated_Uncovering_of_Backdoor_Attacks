import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np


def confusion_matrix_no_grad(data, network, samp=None, device='cpu'):
    """
    Computes confusion matrix and accuracy for a test set.
    :param data: (torch.utils.data.Dataset) test set o be used.
    :param network: (torch.nn.Module) Network to test.
    :param samp: (List[int]) Indices of samples to select.
    :param device: (str) Device to be used ("cpu" or "gpu").
    :return: Accuracy and confusion matrix (numpy.array).
    """
    preds, true = [], []
    dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0, sampler=samp)
    with torch.no_grad():
        network.eval()
        for d, t in dataloader:
            d = d.to(device)
            outputs = network(d.float())
            p = torch.argmax(outputs, 1)
            preds += [p]
            true += [t]
    true, preds = torch.hstack(true).numpy(), torch.hstack(preds).cpu().numpy()
    cm = confusion_matrix(true, preds)
    acc = np.trace(cm) / np.sum(cm)
    return acc, cm


def confusion_matrix_no_grad_protected(data, network, trigger, samp=None, device='cpu', **kwargs):
    """
    Computes confusion matrix and accuracy for a test set with hidden trigger info.
    :param data: (torch.utils.data.Dataset) test set o be used.
    :param network: (torch.nn.Module) Network to test.
    :param trigger: Estimation of hidden trigger.
    :param samp: (List[int]) Indices of samples to select.
    :param device: (str) Device to be used ("cpu" or "gpu").
    :param kwargs: (Dict) Used to specify the kernel_size. Default is 1.
    :return: Accuracy and confusion matrix (numpy.array) after teh defense FL-Bandage.
    """
    step = kwargs.get('kernel_size', 1)
    preds, true = [], []
    dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0, sampler=samp)
    with torch.no_grad():
        network.eval()
        for d, t in dataloader:
            patch = torch.where(trigger != 0, float('nan')*torch.ones_like(d), d)
            check = torch.where(trigger != 0, 1, 0) #1 if not yet modified

            while torch.sum(check) != 0:
                not_nul = torch.where(check != 0)
                set_trigger = set(zip(not_nul[1].tolist(), not_nul[2].tolist()))
                for j in range(len(not_nul[0])):
                    x = not_nul[1][j].item()
                    y = not_nul[2][j].item()
                    neighbors = set(kernel_pixels(x, y, step))-set_trigger
                    neighbors = [i for i in neighbors if 0 <= i[0] < d.shape[-1] and 0 <= i[1] < d.shape[-1]]
                    if neighbors:
                        patch[:, :, x, y] = patch[:, :, [x_[0] for x_ in list(neighbors)], [y_[1] for y_ in list(neighbors)]].mean(dim=2)
                        check[:, x, y] = 0
            patch = patch.to(device)
            outputs = network(patch.float())
            p = torch.argmax(outputs, 1).cpu()
            preds += [p]
            true += [t]

    true, preds = torch.hstack(true).numpy(), torch.hstack(preds).numpy()
    cm = confusion_matrix(true, preds)
    acc = np.trace(cm) / np.sum(cm)
    return acc, cm


def kernel_pixels(x, y, step):
    indices = []
    for i in range(x-step, x+step+1):
        for j in range(y-step, y+step+1):
            indices += [(i, j)]
    return [i for i in indices if (i[0], i[1]) != (x, y)]

