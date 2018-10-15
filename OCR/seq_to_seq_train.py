from . import dataset
from . import utils
from .models import ImageEncoder, MultilayerLSTMCell, Sequence_to_Sequence_Model, Sequence_to_Sequence_Attention_Model

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

import argparse
import random
import os
import time

def parse_false(str):
  return str.lower() != "false"

parser = argparse.ArgumentParser()
parser.add_argument('--attention', required=True, type=parse_false, help='Whether to use attention. False/false are paresed to the boolean False.'
                                                                         'Everything else parses to True.')
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--beam_search', action='store_true', help='Whether to use beam search (default is pointwise prediction)')
parser.add_argument('--teacher_forcing', action='store_true', help='Whether to turn on teacher forcing. Default is teacher forcing is not used.')
parser.add_argument('--workers', type=int, default=5, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden state')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the sequence models')
parser.add_argument('--embedding_size', type=int, default=25, help='size of the embeddings')
parser.add_argument('--alignment_size', type=int, default=100, help='Alignment size used for attention models.')
parser.add_argument('--niter', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for model, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--alphabet', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='List of possible classifications.')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to display loss and validation information.')
parser.add_argument('--n_test_disp', type=int, default=5, help='Number of samples to display when test')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be saved')
parser.add_argument('--rms', action='store_true', help='Whether to use rmsprop (default is adam)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is adam)')
parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset.')
parser.add_argument('--gpu_choice', type=int, default=0, help='Which gpu to use. Default is 0. Multigpu is not currently an option.')
parser.add_argument('--vertical_scale', action='store_true', help='Whether you want the images to be rescaled to a height of 32.')
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

transforms_list = [transforms.Lambda(vertical_scale_preserve_aspect_ratio)] if opt.vertical_scale else []
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
    collate_fn=dataset.collate_by_padding(opt.padding_value))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=transform)

nclass = len(opt.alphabet)
nchannels = 1

converter = utils.StringLabelConverterSeqToSeq(opt.alphabet)
image_encoder = ImageEncoder(nchannels, opt.hidden_size, opt.num_layers, attention=opt.attention)

if opt.attention:
  decoder = MultilayerLSTMCell(opt.embedding_size + 2*opt.hidden_size, opt.hidden_size, opt.num_layers)
  model = Sequence_to_Sequence_Attention_Model(image_encoder, decoder, opt.hidden_size, nclass, opt.embedding_size,
                                               opt.alignment_size)
else:
  decoder = MultilayerLSTMCell(opt.embedding_size, opt.hidden_size, opt.num_layers)
  model = Sequence_to_Sequence_Model(image_encoder, decoder, opt.hidden_size, nclass, opt.embedding_size)

# custom weights initialization called on the image encoder.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

image_encoder.apply(weights_init)

if opt.cuda:
  torch.cuda.set_device(opt.gpu_choice)
  model = model.cuda(opt.gpu_choice)

# setup optimizer
if opt.rms:
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
elif opt.adadelta:
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))

loss_avg = utils.averager()

def trainBatch(net, optimizer):
    images, targets = train_iter.next()
    images, targets = Variable(images), Variable(converter.encode(targets))

    if opt.cuda:
      images, targets = images.cuda(), targets.cuda()

    loss = net.forward_train(images, targets, teacher_forcing=opt.teacher_forcing)

    net.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def val(net, datapoints, max_iter=100):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        datapoints, batch_size=1, pin_memory=True,
        shuffle=True)
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))

    for i in range(max_iter):
        image, target = val_iter.next()
        image = Variable(image)
        target = target[0]

        if opt.cuda:
          image = image.cuda()

        if opt.beam_search:
          prediction = converter.decode(net.beam_search_prediction(image))
        else:
          prediction = converter.decode(net.point_wise_prediction(image))

        if prediction == target:
          n_correct += 1

        if i < opt.n_test_disp:
          print('prediction: %s, gt: %s' % (prediction, target))

    accuracy = n_correct / float(max_iter)
    print('Test accuracy: %f' % accuracy)

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        model.train()

        loss = trainBatch(model, optimizer)
        loss_avg.add(loss)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
            val(model, test_dataset)
            for p in model.parameters():
                p.requires_grad = True

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                model.state_dict(), '{0}/seq_model_{1}_{2}.pth'.format(opt.experiment, epoch, i))

torch.save(model.state_dict(), '{0}/seq_model_final.pth'.format(opt.experiment))
