import json
import zipfile

import PIL.Image
import torch

import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

import glob
import os

import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 use_semantic_labels=False,
                 semantic_label_n_classes=0,
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 cache=False,  # Cache images in CPU memory?
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._use_semantic_labels = use_semantic_labels
        self._semantic_label_n_classes = semantic_label_n_classes
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._cached_semantic_labels = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._raw_semantic_labels = None
        self._label_shape = None
        self._semantic_label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_semantic_labels(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None, _raw_semantic_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        semantic_label = self._cached_semantic_labels.get(raw_idx, None)

        if image is None:
            image, fname = self._load_raw_image(raw_idx)
            if self._use_semantic_labels:
                semantic_label = self._load_raw_semantic_labels(raw_idx)
                assert isinstance(semantic_label, np.ndarray)
                assert list(semantic_label.shape[2:]) == self.image_shape[1:]  # for spatial dimensions
                assert semantic_label.shape[0] == self.image_shape[0]  # for batch dimension
            if self._cache:
                self._cached_images[raw_idx] = image
                if self._use_semantic_labels:
                    self._cached_semantic_labels[raw_idx] = semantic_label
        else:
            image, fname = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx] and self._use_semantic_labels:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
            semantic_label = semantic_label[:, :, ::-1]
            return image.copy(), self.get_label(idx), semantic_label.copy(), fname
        elif self._xflip[idx] and not self._use_semantic_labels:
            return image.copy(), self.get_label(idx), fname
        elif not self._xflip[idx] and self._use_semantic_labels:
            return image.copy(), self.get_label(idx), semantic_label.copy(), fname
        else:
            return image.copy(), self.get_label(idx), fname

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 data_mode='train',  # 'train', 'val', or 'test'
                 output_resolution=None,  # Ensure specific resolution, None = highest available.
                 diffusion_resolution=None,
                 use_pyspng=True,  # Use pyspng if available?
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._data_mode = data_mode
        self._output_resolution = output_resolution
        self._diffusion_resolution = diffusion_resolution  # not used apart from when creating network
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = glob.glob(os.path.join(self._path, '*', '*', '*.png'))
            self._training_fnames = {filepath for filepath in self._all_fnames if
                                     'train' in filepath and 'images' in filepath}
            self._validation_fnames = {filepath for filepath in self._all_fnames if
                                       'val' in filepath and 'images' in filepath}
            self._test_fnames = {filepath for filepath in self._all_fnames if
                                 'test' in filepath and 'images' in filepath}

        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._train_image_fnames = sorted(
            fname for fname in self._training_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._val_image_fnames = sorted(
            fname for fname in self._validation_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._test_image_fnames = sorted(
            fname for fname in self._test_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        if len(self._image_fnames) == 0 or len(self._train_image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        if self._data_mode == 'train':
            raw_shape = [len(self._train_image_fnames)] + list(self._load_raw_image(0)[0].shape)
        elif self._data_mode == 'val':
            raw_shape = [len(self._val_image_fnames)] + list(self._load_raw_image(0)[0].shape)
        elif self._data_mode == 'test':
            raw_shape = [len(self._test_image_fnames)] + list(self._load_raw_image(0)[0].shape)
        else:
            raise ValueError('Invalid data_mode: ' + self._data_mode)

        if output_resolution is not None and (raw_shape[2] != output_resolution or raw_shape[3] != output_resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if self._data_mode == 'train':
            fname = self._train_image_fnames[raw_idx]
        elif self._data_mode == 'val':
            fname = self._val_image_fnames[raw_idx]
        else:
            raise ValueError('Invalid data_mode: ' + self._data_mode)
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = PIL.Image.open(f)
                if self._output_resolution is not None:
                    image = image.resize((self._output_resolution,
                                          self._output_resolution), PIL.Image.NEAREST)
                image = np.array(image)

        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        return image.transpose(2, 0, 1), fname  # HWC => CHW

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_raw_semantic_labels(self, raw_idx):
        if self._data_mode == 'train':
            fname = self._train_image_fnames[raw_idx]
        elif self._data_mode == 'val':
            fname = self._val_image_fnames[raw_idx]
        else:
            raise ValueError('Invalid data_mode: ' + self._data_mode)

        fname = fname.replace('images', 'sector_annotations')
        with self._open_file(fname) as d:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                semantic_label = pyspng.load(d.read())
            else:
                semantic_label = PIL.Image.open(d).convert('L')
                semantic_label = semantic_label.resize((self._output_resolution,
                                                        self._output_resolution),
                                                       PIL.Image.NEAREST)
                semantic_label = np.array(semantic_label)
                semantic_label = np.expand_dims(semantic_label, axis=0)

        semantic_label = torch.tensor(semantic_label).long()

        semantic_label = torch.nn.functional.one_hot(semantic_label, self._semantic_label_n_classes)
        return semantic_label.transpose(1, 3).transpose(2, 3).numpy().astype(np.float32)


class CamusSegmentationDataset(Dataset):

    def __init__(self, image_dir, dataset_mode, data_type):
        self.image_dir = image_dir
        self.dataset_mode = dataset_mode
        self.image_paths = natsorted(glob.glob(os.path.join(self.image_dir, 'images', dataset_mode, '*' + data_type)))
        self.label_paths = natsorted(
            glob.glob(os.path.join(self.image_dir, 'annotations', dataset_mode, '*' + data_type)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_fp = self.image_paths[index]
        out_img = Image.open(image_fp).convert('L')

        label_fp = self.label_paths[index]
        label = Image.open(label_fp).convert('L')

        return np.expand_dims(np.array(out_img), axis=0), np.squeeze(np.array(label))


class CamusClassificationDataset(Dataset):

    def __init__(self, image_dir, dataset_mode, data_type):
        self.image_dir = image_dir
        self.dataset_mode = dataset_mode
        self.image_paths = natsorted(glob.glob(os.path.join(self.image_dir, 'images', dataset_mode, '*' + data_type)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_fp = self.image_paths[index]
        out_img = Image.open(image_fp).convert('L')

        if '2CH' in self.image_paths[index]:
            img_label = 0
        elif '4CH' in self.image_paths[index]:
            img_label = 1
        else:
            raise ValueError

        return np.expand_dims(np.array(out_img), axis=0), np.array(img_label)


class CamusVAEDataset(Dataset):
    # same as CamusSegmentationDataset but with different label paths (sector_annotations vs annotations)
    def __init__(self, image_dir, dataset_mode, data_type):
        self.image_dir = image_dir
        self.dataset_mode = dataset_mode
        self.image_paths = natsorted(glob.glob(os.path.join(self.image_dir, 'images', dataset_mode, '*' + data_type)))
        self.label_paths = natsorted(
            glob.glob(os.path.join(self.image_dir, 'sector_annotations', dataset_mode, '*' + data_type)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_fp = self.image_paths[index]
        out_img = Image.open(image_fp).convert('L')

        label_fp = self.label_paths[index]
        label = Image.open(label_fp).convert('L')

        return np.expand_dims(np.array(out_img), axis=0), np.squeeze(np.array(label))
