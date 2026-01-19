# coding=utf-8
import os
import random
import torch
import numpy as np
import scipy.io as io
from torchvision import datasets, transforms
from .image_augment import CIFAR10Policy, Cutout, ImageNetPolicy


def random_spilt(root_dir, train_radio):
    train_items = []
    test_items = []
    for (root, dirs, files) in os.walk(root_dir):
        dirs.sort()

        shuffled_indices = np.random.permutation(len(files))
        train_size = int(len(files) * train_radio)
        train_indices = shuffled_indices[:train_size]
        # test_indices = shuffled_indices[train_size:]
        for i, f in enumerate(files):
            if f.endswith("png") or f.endswith("mat"):
                r = root.split('/')
                lr = len(r)
                if i in train_indices:
                    train_items.append((f, r[lr - 2] + "/" + r[lr - 1], root))
                else:
                    test_items.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== TrainSet %d items, TestSet %d items" % (len(train_items), len(test_items)))
    return train_items, test_items


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class Event2Frame_RCS(object):
    """ Convert DVS event streams to frames
    Args:
    """

    def __init__(self, img_size, tr, ts):
        self.img_size = img_size
        self.tr = tr  # time resolution
        self.ts = ts  # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([2, self.img_size[0], self.img_size[1], self.ts], dtype=int)  # frame
        maxT = np.max(events['TD'].ts)
        length = int(self.ts * self.tr)
        start = random.randrange(0, maxT - length - 1, self.tr)
        end = start + length
        for j in range(start, end, int(self.tr)):  # frames with tr ms
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 2)
            frame[0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1, int((j - start) / self.tr)] = 1.0
            frame[1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1, int((j - start) / self.tr)] = 1.0
            # im = Image.fromarray((frame[0, ..., int((j-start) / self.tr)] * 255).astype(np.uint8), "L")  # numpy to PIL Image
            # im.save(os.path.join('./', str(j / self.tr) + '.png'))
        return np.reshape(frame, (2, self.img_size[0], self.img_size[1], self.ts))


class Event2Frame(object):
    """ Convert DVS event streams to frames
    Args:
    """

    def __init__(self, img_size, tr, ts):
        self.img_size = img_size
        self.tr = tr  # time resolution
        self.ts = ts  # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([2, self.img_size[0], self.img_size[1], self.ts], dtype=int)  # frame
        for j in range(0, int(self.ts * self.tr), int(self.tr)):  # frames with tr ms
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 2)
            frame[0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1, int(j / self.tr)] = 1.0
            frame[1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1, int(j / self.tr)] = 1.0
        #     im = Image.fromarray((frame[int(j / self.tr), :] * 255).astype(np.uint8), "L")  # numpy to PIL Image
        #     im.save(os.path.join('./', str(j / self.tr) + '.png'))
        return np.reshape(frame, (2, self.img_size[0], self.img_size[1], self.ts))


class Event2Frame_FULL(object):
    """ Convert DVS full event streams to frames
    Args:
    """

    def __init__(self, input_shape, ts):
        self.input_shape = input_shape
        self.ts = ts  # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([self.ts, self.input_shape[0], self.input_shape[1], self.input_shape[2]],
                         dtype=np.float32)  # frame
        tr = events['TD'].ts[-1] // self.ts
        for j in range(0, int(self.ts * tr), int(tr)):  # frames with tr ms
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + tr) & (events['TD'].p == 2)
            frame[int(j / tr), 0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1] = 1.0
            frame[int(j / tr), 1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1] = 1.0
        return frame


class CIFAR10_DVS(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 transform=None,
                 target_transform=None):
        self.all_items = data_list
        self.transform = transform
        self.target_transform = target_transform
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        classname = self.all_items[index][1]
        filepath = self.all_items[index][2]

        img = os.path.join(str(filepath), str(filename))
        target = self.idx_classes[classname]
        # print(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)


class CIFAR10_DVS_Aug(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # 1. Load data
        data, target = torch.load(self.root + '/{}.pt'.format(index),weights_only=True)
        
        # 2. Ensure data is float and normalized to [0, 1]
        # Data on disk is already uint8 [0-255], just divide by 255
        if data.dtype == torch.uint8:
            data = data.float() / 255.0
        elif data.max() > 1.0:
            data = data.float() / 255.0
            
        # 3. Resize (Using native torch interpolation to avoid PIL issues)
        # Input needs to be (N, C, H, W) or (C, H, W), interpolate needs (N, C, H, W)
        T, C, H, W = data.shape
        # Here data is (T, 2, H, W)
        data = torch.nn.functional.interpolate(data, size=(48, 48), mode='bilinear', align_corners=False)
        
        # 4. Augmentation (Flip, Roll)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def load_dataset(name, root, cutout=False, auto_aug=False, time_step=10):
    num_class, normalize, train_data, test_data = None, None, None, None
    train_transform = []
    if name == 'CIFAR10' or name == 'CIFAR100':
        train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        train_transform.append(CIFAR10Policy())
    train_transform.append(transforms.ToTensor())
    if cutout:
        train_transform.append(Cutout(n_holes=1, length=16))
    if name == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        num_class = 10
    elif name == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        num_class = 100
    elif name == 'MNIST':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        num_class = 10
    train_transform.append(normalize)
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize
                                        ])
    if name == 'CIFAR100':
        train_data = datasets.CIFAR100(root=root, train=True, download=True,
                                       transform=train_transform)
        val_data = datasets.CIFAR100(root=root, train=False, download=True,
                                     transform=val_transform)
    elif name == 'CIFAR10':
        train_data = datasets.CIFAR10(root=root, train=True, download=True,
                                      transform=train_transform)
        val_data = datasets.CIFAR10(root=root, train=False, download=True,
                                    transform=val_transform)
    elif name == 'MNIST':
        train_data = datasets.MNIST(root=root, train=True, download=True,
                                    transform=train_transform)
        val_data = datasets.MNIST(root=root, train=False, download=True,
                                  transform=val_transform)
    elif name == 'imagenet':
        num_class = 1000

        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if auto_aug:
            train_transform_list.append(ImageNetPolicy())
        
        train_transform_list.append(transforms.ToTensor())
        
        if cutout:
            train_transform_list.append(Cutout(n_holes=1, length=112))

        train_transform_list.append(normalize)
        
        transform_train = transforms.Compose(train_transform_list)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_data = datasets.ImageFolder(root=traindir, transform=transform_train)
        val_data = datasets.ImageFolder(root=valdir, transform=transform_test)
    elif name == 'CIFAR10_DVS' or name == 'CIFAR10_DVS_SJ':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        
        def check_and_clean_dvs(root, time_step):
            import shutil
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            paths_to_check = [
                os.path.join(root, 'events_np'),
                os.path.join(root, f'frames_number_{time_step}_split_by_number')
            ]
            for p in paths_to_check:
                if os.path.exists(p):
                    should_delete = False
                    for c in classes:
                        c_path = os.path.join(p, c)
                        if not os.path.exists(c_path) or not os.listdir(c_path):
                            should_delete = True
                            break
                    if should_delete:
                        shutil.rmtree(p)

        check_and_clean_dvs(root, time_step)
        
        # Simplified augmentation
        resize_op = transforms.Resize(size=(48, 48))
        def trans_train(data):
            data = torch.from_numpy(data).float()
            data = resize_op(data)
            if random.random() > 0.5:
                data = torch.flip(data, dims=(3,))
            return data

        def trans_test(data):
            data = torch.from_numpy(data).float()
            return resize_op(data)

        data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=time_step, split_by='number', transform=trans_train)
        train_data, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))

        data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=time_step, split_by='number', transform=trans_test)
        _, val_data = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
        num_class = 10
    else:
        raise NotImplementedError()
    return train_data, val_data, num_class
