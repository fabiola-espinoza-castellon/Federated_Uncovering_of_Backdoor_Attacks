# FUBA: Federated Uncovering of Backdoor Attacks

# Before usage
This section lists any major frameworks/libraries used in the code.
* matplotlib==3.2.2
* numpy==1.24.1
* Pillow==9.4.0
* scikit_learn==1.2.0
* scipy==1.5.0
* torch==1.12.1
* torchvision==0.13.1

Note that if you want to use FEMNIST dataset, you should have a the matlab file emnist-digits.mat It can be downloaded from [here](https://www.nist.gov/itl/products-and-services/emnist-dataset)[1].

[1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.

# Usage
Call function main.py with python followed by the arguments of your choice. They are :
  * (int) nb_users: number of users to be created.
  * (int) attack_target: attack target.
  * (str) attack_type: type of attack. Supported options are 'square', 'cross' and 'copyright' for MNIST, FashionMNIST and FEMNIST datasets, and 'yellow sticker' for GTSRB.
  * (int) nb_attack: number of malicious clients.
  * (float) p_attack: proportion of poised local dataset for malicious clients.
  * (str) dataset: dataset to be used. Supported options are 'MNIST', 'FashionMNIST', 'FEMNIST' and 'GTSBR'.
  * (int) rounds: number of FL rounds.
  * (int) E: number of local epochs.
  * (int) B: local batch size.
  * (float) lr: local learning rate.
  * (float) C: proportion of clients sampled at each round.
  * (int) samples: number of random samples in FL-Bandage.
  * (int) iterations: number of iterations for trigger estimation.
  * (float) lr_bandage: learning rate in FL-Bandage.
  * (float) gamma: proportion of estimated trigger to be kept (gamma in FL-Bandage).
  * (int) label_skew: (Optional) If label skewed distribution (non-IID), number of labels per client. Default is False.
  * (int) kernel_size: (Optional) Kernel size in FL-Bandage. Default is 1. 1 -> 3x3 kernel, 2->5x5 kernel.
  * (str) data_path: (Optional) Path where dataset is stored or will be downloaded. For dataset 'FEMNIST', file 'emnist-digits.mat' must be located in file. Default is current file.
  * (str) server_preload_net: (Optional) Path to .pth file if global model already trained.
  * (bool) save: (Optional) Save results (.pth) after FL training and after defense. Default is False.
  * (bool) cuda: (Optional) True to use option to computed in GPU. Default is False.


We can image the following case : 100 total users under a "cross" attack with target 5 for FashionMNIST classification.
The distribution between clients is label skewed: each client only has 5 labels.
10 malicious users poise 0.5 of their data. During 100 FL rounds, all clients participate (C=1) with E=5 local epochs, B=50 local batch size and lr=0.01 learning rate.

For defense, 50 samples and 2 iterations were used to recreate the hidden trigger. The bandage learning rate was 0.1 and only gamma=0.1 of the reconstructed information was kept.
We decide to save the results and to use a GPU to compute this case. 

The corresponding command line would be :
```bash
python main.py  100 5 cross 10 0.5 FashionMNIST 100 5 50 0.01 1 50 2 0.1 0.1 5 --save --cuda

```
