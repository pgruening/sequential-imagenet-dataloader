# dataloader respecting the PyTorch conventions, but using tensorpack to load and process

import os

import cv2
import torch

import numpy as np
<<<<<<< HEAD
import tensorpack.dataflow as td
from tensorpack import imgaug
from tensorpack.dataflow import (AugmentImageComponent, PrefetchDataZMQ,
                                 BatchData, MultiThreadMapData)

#####################################################################################################
# copied from: https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/imagenet_utils.py #
#####################################################################################################


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """

    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(
                self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(
                    out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(
            self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors
#####################################################################################################
#####################################################################################################


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class Loader(object):
=======
import dataflow as td
from io import BytesIO
from PIL import Image


class ImagenetLoader(object):
>>>>>>> 911f4e9b2de593ed770802dd1e68e11761b5b1eb
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 4)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

<<<<<<< HEAD
    def __init__(self, mode, batch_size=256, shuffle=False, num_workers=25, cache=50000,
                 collate_fn=default_collate, drop_last=False, cuda=False):
=======
    def __init__(self, imagenet_dir, mode, transform, batch_size, shuffle=False, num_workers=4, cache=50000,
            drop_last=False):
        if drop_last:
            raise NotImplementedError("drop_last not implemented")
>>>>>>> 911f4e9b2de593ed770802dd1e68e11761b5b1eb
        # enumerate standard imagenet augmentors
        assert mode in ['train', 'val'], mode

<<<<<<< HEAD
        # load the lmdb if we can find it
        lmdb_loc = os.path.join(
            os.environ['IMAGENET'], 'ILSVRC-%s.lmdb' % mode)
        ds = td.LMDBData(lmdb_loc, shuffle=False)
        if shuffle:
            ds = td.LocallyShuffleData(ds, cache)
        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.LMDBDataPoint(ds)
        ds = td.MapDataComponent(
            ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        ds = td.AugmentImageComponent(ds, imagenet_augmentors)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()
=======
        # open the lmdb file
        lmdb_loc = os.path.join(imagenet_dir, 'ILSVRC-%s.lmdb'%mode)
        ds = td.LMDBData(lmdb_loc, shuffle=False)
        if shuffle:
            ds = td.LocallyShuffleData(ds, cache)
        def f(x):
            img, label= td.LMDBSerializer._deserialize_lmdb(x)
            # img, label = x
            img = Image.open(BytesIO(img.tobytes())).convert('RGB')
            img = transform(img)
            return img, label
        # ds = td.MultiProcessMapDataZMQ(ds, num_proc=num_workers, map_func=f)
        ds = td.MultiThreadMapData(ds, num_thread=num_workers, map_func=f)
        # ds = td.MapData(ds, f)
        self.ds = td.BatchData(ds, batch_size, use_list=True, remainder=False)
        # self.ds.reset_state()
>>>>>>> 911f4e9b2de593ed770802dd1e68e11761b5b1eb

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds.reset_state()
        self.ds_iter = iter(self.ds)
        self.N = self.ds.size()
        self.i = 0

    def __iter__(self):
<<<<<<< HEAD
        for x, y in self.ds.get_data():
            if self.cuda:
                # images come out as uint8, which are faster to copy onto the gpu
                x = torch.ByteTensor(x).cuda()
                y = torch.IntTensor(y).cuda()
                # but once they're on the gpu, we'll need them in
                yield uint8_to_float(x), y.long()
            else:
                yield uint8_to_float(torch.ByteTensor(x)), torch.IntTensor(y).long()
=======
        self.i = 0
        return self
>>>>>>> 911f4e9b2de593ed770802dd1e68e11761b5b1eb

    def __next__(self):
        if (self.i + 1) == self.N:
            raise StopIteration
        x, y = next(self.ds_iter)
        self.i += 1
        x, y = torch.stack(x), torch.tensor(y)
        return x, y

<<<<<<< HEAD

def uint8_to_float(x):
    x = x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)
    return x.float() / 128. - 1.

=======
    def __len__(self):
        return self.N
>>>>>>> 911f4e9b2de593ed770802dd1e68e11761b5b1eb

if __name__ == '__main__':
    from tqdm import tqdm
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dl = ImagenetLoader(os.environ['DBLOC'], 'train', transform, 256, num_workers=4, shuffle=True)
    # td.TestDataSpeed(dl.ds).start()
    for x, y in tqdm(dl, total=len(dl)):
        x, y = x.cuda(), y.cuda()

