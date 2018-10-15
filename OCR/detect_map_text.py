import torch
import torchvision.transforms as transforms

from . import models
from . import dataset
from . import utils

from ..Util.decorators import timeit
from ..Util.log import log

from torch.autograd import Variable

from pathlib import Path
from functools import partial
from operator import attrgetter

from .predict_street import predict_street
from .model_functions import predict_digit, predict_word

def find_text_images(folder_path):
    for file in Path(folder_path).glob("*rot_word*.png"):
        if "st" in str(file) or "hn" in str(file):
            yield file

def predict_text_given_path(model, transform, converter, image_path):
    image = dataset.greyscale_image_loader(str(image_path))
    return predict_text(model, transform, converter, image)

def predict_text(model, transform, converter, np_image):
    image = dataset.numpy_image_loader(np_image)

    if transform is not None:
        image = transform(image)

    datapoint = Variable(image.unsqueeze(0))

    prediction = model(datapoint).squeeze(1)
    prediction_size = Variable(torch.IntTensor([prediction.size(0)]))

    _, prediction = prediction.max(1)
    sim_prediction = converter.decode(prediction.data, prediction_size.data, raw=False)[0]

    return sim_prediction


@timeit
def detect_map_text(street_numbers, house_numbers, cuda=False):
    # digit_model = loadDigitModel("./src/OCR/models/sanborn_digit_model.pth", upside_down=False, cuda=cuda)
    # upside_down_model = loadDigitModel("./src/OCR/models/upside_down_digit_model.pth", upside_down=True, cuda=cuda)
    # number_model = loadSequenceModel("./src/OCR/models/number_model.pth", numbers_only=True, cuda=cuda)
    # word_model = loadSequenceModel("./src/OCR/models/word_model.pth", numbers_only=False, cuda=cuda)

    return predict_street(street_numbers, house_numbers, cuda=cuda)
