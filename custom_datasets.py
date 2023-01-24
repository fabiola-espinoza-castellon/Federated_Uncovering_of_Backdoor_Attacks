import csv
import pathlib
from scipy.io import loadmat
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class GTSRBcustom(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.targets = torch.tensor([sample[1] for sample in self._samples])
        self.data = [PILToTensor()(Image.open(path[0]).convert("RGB").resize((32, 32), Image.BILINEAR)) for path in self._samples]
        self.data = torch.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.permute((0, 2, 3, 1))  # convert to HWC

        self.class_to_idx = {'Speed limit (20km/h)': 0,
                             'Speed limit (30km/h)': 1,
                             'Speed limit (50km/h)': 2,
                             'Speed limit (60km/h)': 3,
                             'Speed limit (70km/h)': 4,
                             'Speed limit (80km/h)': 5,
                             'End of speed limit (80km/h)': 6,
                             'Speed limit (100km/h)': 7,
                             'Speed limit (120km/h)': 8,
                             'No passing': 9,
                             'No passing veh over 3.5 tons': 10,
                             'Right-of-way at intersection': 11,
                             'Priority road': 12,
                             'Yield': 13,
                             'Stop': 14,
                             'No vehicles': 15,
                             'Veh > 3.5 tons prohibited': 16,
                             'No entry': 17,
                             'General caution': 18,
                             'Dangerous curve left': 19,
                             'Dangerous curve right': 20,
                             'Double curve': 21,
                             'Bumpy road': 22,
                             'Slippery road': 23,
                             'Road narrows on the right': 24,
                             'Road work': 25,
                             'Traffic signals': 26,
                             'Pedestrians': 27,
                             'Children crossing': 28,
                             'Bicycles crossing': 29,
                             'Beware of ice/snow': 30,
                             'Wild animals crossing': 31,
                             'End speed + passing limits': 32,
                             'Turn right ahead': 33,
                             'Turn left ahead': 34,
                             'Ahead only': 35,
                             'Go straight or right': 36,
                             'Go straight or left': 37,
                             'Keep right': 38,
                             'Keep left': 39,
                             'Roundabout mandatory': 40,
                             'End of no passing': 41,
                             'End no passing veh > 3.5 tons': 42}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Check : https://www.kaggle.com/code/shivank856/gtsrb-cnn-98-test-accuracy

        img, target = self.data[index], self.targets[index]

        img = img.permute(2, 1, 0)/255

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self._samples)

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )


class FEMNISTcustom():
    """
     Class used to load and preprocess the FEMNIST dataset.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        """
        Set up the dataset.
        :param root: (string) Root directory for the dataset that must contain emnist-digits.mat file. Automatic
         download has not yet been implemented.
        :param train: (boolean) True if dataset used for training, else False.
        :param transform: (callable) Optional tensor transform.
        :param target_transform: (callable) Optional target transform callables.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        d = loadmat(self.root)['dataset'][0][0]
        if self.train:
            i = 'train'
        else:
            i = 'test'

        self.data = torch.Tensor(d[i]['images'][0][0].reshape(-1, 28, 28).transpose((0, 2, 1)))
        self.targets = torch.Tensor(d[i]['labels'][0][0].reshape(-1)).type(torch.int64)
        self.writers = d[i]['writers'][0][0].reshape(-1)
        self.class_to_idx = {k: k for k in range(self.targets.unique().max().item()+1)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index].unsqueeze(0)/255, self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)


class UniformSamePixelsData(Dataset):
    """
    Creates a dataset of tensors of specified shape where each tensor has the same value in all its pixels and the
    target is a scalar value specified in target.
    Attributes:
        target (tensor): Tensor of targets.
        data (tensor): Tensor of uniform data.
    """

    def __init__(self, shape, target):
        """
        Initializes class and creates dataset.
        :param shape: (tuple[int]) Tuple of integers representing the shape of the data tensor.
        :param target: (int) Target label for all the data points.
        """
        super().__init__()
        self.target = target*torch.ones(shape[0], dtype=torch.long)
        self.data = torch.ones(shape)
        values = torch.rand(shape[0])
        for i in range(len(values)):
            self.data[i] = values[i] * self.data[i]

    def __getitem__(self, index: int):
        return self.data[index], self.target[index]

    def __len__(self) -> int:
        return len(self.data)

