import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from src.Util.log import log
from src.Util import constants as const
from .models import ResidualNet, CRNN
from .dataset import tightest_image_crop, square_padding, vertical_scale_preserve_aspect_ratio
from . import utils



def load_OCR_models():
    """ Load all relevant OCR models, if they haven't been loaded before """
    if const.DIGIT_MODEL == 0:  # models haven't been loaded yet
        log.debug("Loading OCR models")
        const.DIGIT_MODEL = load_digit_model("./src/OCR/models/sanborn_digit_model.pth", upside_down=False, cuda=const.CUDA)
        const.UPSIDE_DOWN_MODEL = load_digit_model("./src/OCR/models/upside_down_digit_model.pth", upside_down=True, cuda=const.CUDA)
        const.NUMBER_MODEL = load_sequence_model("./src/OCR/models/number_model.pth", numbers_only=True, cuda=const.CUDA)
        const.WORD_MODEL = load_sequence_model("./src/OCR/models/word_model.pth", numbers_only=False, cuda=const.CUDA)
        log.debug("Done loading OCR models")
    else:
        log.debug("OCR models already loaded")

def load_digit_model(model_path, upside_down, cuda=False):
    """ Takes in the path to a digit model (model_path) as a string and a boolean that determines
        whether or not the model contains classes for upside down digits (upside_down). Returns the
        loaded model.
    """
	  # 10 classes for regular model, 13 for upside down model
    n_class = 10
    if upside_down:
        n_class = 13
    n_channels = 1
    height = 32
    width = 32
    n_blocks = 3

    model = ResidualNet(n_channels, height, width, n_blocks, n_class)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    if cuda:
        model.cuda()

    return model


def load_sequence_model(model_path, numbers_only=True, cuda=False):
    """ Takes in the path to a full sequence model (model_path) as a string and returns the loaded
        model.
    """

    alphabet = '0123456789' if numbers_only else '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n_class = len(alphabet) + 1
    n_channels = 1
    n_lstm_hidden_size = 256

    number_model = CRNN(n_channels, n_class, n_lstm_hidden_size)
    number_model.load_state_dict(torch.load(model_path))
    number_model.train(False)

    if cuda:
        number_model.cuda()

    return number_model


def predict_digit(image, model, cuda=False):
    """ Takes in a model and a pillow image and predicts what digit is in the image using the model.
        Boolean parameter for cuda included as well.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(tightest_image_crop),
        transforms.Lambda(square_padding),
        transforms.ToPILImage(),
        transforms.Resize(32, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)

    if cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    output = model(image)
    _, pred = torch.max(output.data, 1)
    pred = pred[0]

    return pred

def predict_word(image, model, number_only, cuda=False):
    """ Takes in an image of a word, a model, a boolean option to say if it is a number only model
    or not, and a boolean for cuda. Uses the model to predict what the word in the image says and
    returns that prediction.
    """
    transform = transforms.Compose([transforms.Lambda(vertical_scale_preserve_aspect_ratio),
                                    transforms.ToTensor()])

    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if number_only:
        alphabet = '0123456789'
    converter = utils.strLabelConverter(alphabet)

    image = Image.fromarray(image)
    image = transform(image)

    if cuda:
        datapoint = Variable(image.unsqueeze(0).cuda())
    else:
        datapoint = Variable(image.unsqueeze(0))

    pre_pred = model(datapoint)

    prediction = pre_pred.squeeze(1)
    prediction_size = Variable(torch.IntTensor([prediction.size(0)]))

    _, prediction = prediction.max(1)
    sim_prediction = converter.decode(prediction.data, prediction_size.data, raw=False)[0]

    return sim_prediction
