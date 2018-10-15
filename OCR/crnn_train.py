from .models import CRNN
from . import dataset
from . import utils

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from warpctc_pytorch import CTCLoss
from PIL import Image

import os
import time
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--nhidden', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for model, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='List of possible classifications.')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to display loss and validation information.')
parser.add_argument('--n_test_disp', type=int, default=5, help='Number of samples to display when test')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be saved')
parser.add_argument('--rms', action='store_true', help='Whether to use rmsprop (default is adam)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is adam)')
parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset.')
parser.add_argument('--gpu_choice', type=int, default=0, help='Which gpu to use. Default is 0. Multigpu is not currently an option.')
parser.add_argument('--vertical_scale', action='store_true', help='Whether you want the images to be rescaled to a height of 32. Default is not to.')
parser.add_argument('--padding_value', type=int, default=0, help='What value to pad with when making batches. Default is 0.')
parser.add_argument('--normalize', action='store_true', help='Whether you want to normalize the images. Default is not to.')
parser.add_argument('--deskew_constant', type=float, default=0, help='What amount to deskew the data by. If 0 image will not be deskewed.'
                                                                     'Default is not to.')
parser.add_argument('--random_deskew', action='store_true', help='Whether to randomly choose the deskew value. If this is set, then deskew constant'
                                                                 'should also be set to a positive integer. It will then consider each multiple of'
                                                                 ' 0.1 up to the deskew constant/10 (exclusive) with equal probability.')
parser.add_argument('--use_masking', action='store_true', help='Whether to mask the parts of the image that arise from being padded.')

opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transforms_list = [transforms.Lambda(dataset.vertical_scale_preserve_aspect_ratio)] if opt.vertical_scale else []
if opt.deskew_constant != 0:
  if opt.random_deskew:
    transforms_list.append(transforms.Lambda(lambda point: dataset.deskew(random.randrange(opt.deskew_constant)/10, point)))
  else:
    transforms_list.append(transforms.Lambda(lambda point: dataset.deskew(opt.deskew_constant, point)))
transforms_list.append(transforms.ToTensor())
if opt.normalize:
  transforms_list.append(transforms.Normalize((0.5,), (0.5,)))

transform = transforms.Compose(transforms_list)

train_dataset = dataset.lmdbDataset(root=opt.trainroot, transform=transform)
assert train_dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batch_size,
    shuffle=opt.shuffle, num_workers=opt.workers, pin_memory=True,
    collate_fn=dataset.collate_by_padding(opt.padding_value, with_widths=opt.use_masking))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=transform)

nclass = len(opt.alphabet) + 1
nchannels = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = CRNN(nchannels, nclass, opt.nhidden)
crnn.apply(weights_init)

image = torch.FloatTensor(opt.batch_size, 1, 1, 1)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)


if opt.cuda:
    torch.cuda.set_device(opt.gpu_choice)
    crnn = crnn.cuda(opt.gpu_choice)
    image = image.cuda(opt.gpu_choice)
    criterion = criterion.cuda(opt.gpu_choice)

if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.rms:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))

def val(net, datapoints, criterion, max_iter=5):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        datapoints, batch_size=opt.batch_size, pin_memory=True, num_workers=opt.workers, shuffle=True,
        collate_fn=dataset.collate_by_padding(opt.padding_value, with_widths=opt.use_masking))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1

        if opt.use_masking:
          (cpu_images, widths), cpu_texts = data
        else:
          cpu_images, cpu_texts = data

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        if opt.use_masking:
          preds = net(image, widths=widths)
        else:
          preds = net(image)

        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)

    for raw_pred, pred, gt, _ in zip(raw_preds, sim_preds, cpu_texts, range(opt.n_test_disp)):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batch_size)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    if opt.use_masking:
      (cpu_images, widths), cpu_texts = data
    else:
      cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)

    t, l = converter.encode(cpu_texts)

    utils.loadData(text, t)
    utils.loadData(length, l)

    if opt.use_masking:
      preds = net(image, widths=widths)
    else:
      preds = net(image, widths=widths)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    cost = criterion(preds, text, preds_size, length) / batch_size

    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
            val(crnn, test_dataset, criterion)
            for p in crnn.parameters():
                p.requires_grad = True

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))

torch.save(crnn.state_dict(), '{0}/netCRNN_final.pth'.format(opt.experiment))
