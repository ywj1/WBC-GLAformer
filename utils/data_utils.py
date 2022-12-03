import logging
from PIL import Image
import os

import torch

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, WeightedRandomSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                              transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                             # transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               # cleaned=os.path.join(data_dir,'cleaned.dat'),
                               transform=transforms.Compose([
                                   transforms.Resize((600, 600), Image.BILINEAR),
                                   transforms.RandomCrop((448, 448)),
                                   transforms.RandomHorizontalFlip(),
                                   AutoAugImageNetPolicy(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                               )
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                              transform=transforms.Compose([
                                  transforms.Resize((600, 600), Image.BILINEAR),
                                  transforms.CenterCrop((448, 448)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
    elif args.dataset == 'dog':
        train_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                              transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                             transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False
                        )
        testset = dogs(root=args.data_root,
                       train=False,
                       cropped=False,
                       transform=test_transform,
                       download=False
                       )
    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform = transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                              # transforms.RandomCrop((304, 304)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                             # transforms.CenterCrop((304, 304)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'blood_cell_23':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.648, 0.515, 0.682], [0.184, 0.251, 0.145])])
        # transforms.RandomErasing(),
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.648, 0.516, 0.681], [0.185, 0.251, 0.146])])

        trainset = datasets.ImageFolder('/home/vipuser/Mywork/blood_cell_23/train',
                                        transform=transform_train)
        testset = datasets.ImageFolder('/home/vipuser/Mywork/blood_cell_23/val',
                                       transform=transform_test)

    elif args.dataset == 'PBC':
        transform_train = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(hue=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.6481272, 0.5147434, 0.68180877], [0.18407342, 0.25081202, 0.1447535])])
        # transforms.RandomErasing(),
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.6477913, 0.51561946, 0.6812609], [0.1848757, 0.25133228, 0.14552529])])

        trainset = datasets.ImageFolder('/home/workspace/qin/chenben/Data/PBC/val',
                                        transform=transform_train)
        testset = datasets.ImageFolder('/home/workspace/qin/chenben/Data/PBC/val',
                                       transform=transform_test)
    elif args.dataset == 'ImageNet-1K':
        train_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                              transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              # AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                             transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = datasets.ImageFolder('/mnt/hangzhou_116_homes/cl/datasets/ILSVRC2012/train', transform=train_transform)
        testset = datasets.ImageFolder('/mnt/hangzhou_116_homes/cl/datasets/ILSVRC2012/test', transform=test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # def make_weights_for_balanced_classes(images, nclasses):
    #     count = [0] * nclasses
    #     for item in images:
    #         count[item[1]] += 1
    #     weight_per_class = [0.] * nclasses
    #     N = float(sum(count))
    #     for i in range(nclasses):
    #         weight_per_class[i] = N / float(count[i])
    #     weight = [0] * len(images)
    #     for idx, val in enumerate(images):
    #         weight[idx] = weight_per_class[val[1]]
    #     return weight
    #
    # weights = make_weights_for_balanced_classes(trainset.imgs, len(
    #     trainset.classes))  # 获取定义的权重
    # weights = torch.DoubleTensor(weights)

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

    test_sampler = RandomSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)

    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
