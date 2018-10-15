import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F

import cv2
import lmdb
import six
from PIL import Image
import numpy as np

import sys
import random
from functools import partial

# Create an lmdb dataset with our dataset of images and their labels. The training process for our neural network requires the data to be in an lmdb dataset.
class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("The index given was outside of the possible bounds.")

        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)

class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

# We need all of the images in a batch to have the same width. We find the maximum width in a batch and pad all of the other images with zeros so that all images in the batch have the same width.
def pad_to_width(desired_width, padding_value, image):
    _, height, width = image.size()

    if width == desired_width:
        return image

    padding = torch.FloatTensor([[[padding_value]]]).repeat(1, height, desired_width-width)
    return torch.cat((image, padding), dim=2)

def square_padding(img):
    _, height, width = img.size()

    if height > width:
        padding = torch.zeros(1, height, height-width)
        return torch.cat((padding, img), 2)
    elif width > height:
        padding = torch.zeros(1, width-height, width)
        return torch.cat((padding, img), 1)

    return img

# Pads all images in a batch to be the same length. Takes in a batch of images and returns the same batch padded to the same length.
class collate_by_padding(object):
    def __init__(self, padding_value, with_widths=False):
        self.padding_value = padding_value
        self.with_widths = with_widths

    def __call__(self, batch):
        images, labels = zip(*batch)
        widths = torch.LongTensor(list(map(lambda image: image.size()[2], images)))
        max_width = max(widths)
        padded_images = list(map(partial(pad_to_width, max_width, self.padding_value), images))
        images = torch.stack(padded_images)

        if self.with_widths:
          return (images, widths), labels
        else:
          return images, labels

class AdvancedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, filter_fn=None, shuffle=False):
        super(AdvancedImageFolder, self).__init__(root, transform, target_transform, loader)

        if filter_fn is not None:
            self.imgs = list(filter(filter_fn, self.imgs))

        if shuffle:
            random.shuffle(self.imgs)

# Takes in a path to an image and converts to greyscale.
def greyscale_image_loader(path):
    with open(path, 'rb') as file:
        with Image.open(file) as img:
            return img.convert('L')

def numpy_image_loader(np_image):
    """ Reads a PIL image from a numpy matrix """
    return Image.fromarray(np_image)

# Takes in an image and crops it to the tightest bounding rectangle around the text within the image.
def tightest_image_crop(img, preserve_aspect_ratio=False):
    image_indices = F.threshold(torch.autograd.Variable(img[0]), 0.0000001, 0).data.nonzero()
    top_i = image_indices[0,0]
    bottom_i = image_indices[-1,0]

    mins, _ = image_indices.min(dim=0)
    left_i = mins[1]

    maxs, _ = image_indices.max(dim=0)
    right_i = maxs[1]

    new_width = right_i-left_i+1
    new_height = top_i-bottom_i+1

    if preserve_aspect_ratio:
        if new_width > new_height:
            result = img[:, top_i:top_i+new_width, left_i:right_i+1]
            return img[:, top_i:top_i+new_width, left_i:right_i+1]
        else:
            result = img[:, top_i:bottom_i+1, left_i:left_i+new_height]
            return img[:, top_i:bottom_i+1, left_i:left_i+new_height]

    return img[:, top_i:bottom_i+1, left_i:right_i+1]

# Takes in an image and scales its height to a specified value while preserving the original aspect ratio.
def vertical_scale_preserve_aspect_ratio(img, height=32):
    w, h = img.size

    return transforms.Resize((height, int(height/h * w)), interpolation=Image.NEAREST)(img)

def deskew(deskew_constant, pil_img):
  img = np.asarray(pil_img)
  h, w = img.shape
  deskew_matrix = np.array([[1, deskew_constant, -32*deskew_constant/2], [0, 1, 0]])
  return  Image.fromarray(cv2.warpAffine(img, deskew_matrix, (w,h), flags=cv2.INTER_NEAREST))
