from .model_functions import predict_digit, predict_word
from ..Util.read_data import OCR_image
from . import dataset
import src.Util.constants as const
import numpy as np
from scipy.stats import entropy

def predict_street(street_names, house_numbers, cuda=False):
    """ Takes in a dictionary of street names, a dictionary of house numbers
        where all numbers in a dictionary entry are associated with the same
        street segment, all four of our models, and an option for cuda. Tries
        to predict the orientation of all the house numbers on a street. If
        there are less than 10 house numbers on a street or the majority of
        those house numbers are length 1 or 2, then the orientation with the
        fewest upside down 3s, 4s, or 7s predicted by the upside down model is
        used as the correct orientaion (Method 1). Otherwise, the words are
        predicted using normal models with both orientations. The frequency of
        each number in the first character slot is calculated and the orientation
        that has frequencies with the lowest entropy is predicted to be the correct
        orientaion (Method 2). The street names are processed in whatever orientation
        they are passed in with.
    """
    # predicting street names
    predictions = pred_house_numbers(house_numbers, cuda=cuda)

    # Predicting street names
    pred_street_names(street_names, predictions, cuda=cuda)

    return predictions

def pred_house_numbers(house_numbers, cuda=False):
    predictions = {}
    for key in house_numbers:
        word_list = house_numbers[key]

        # Method 1 (see doc string)
        if use_method1(word_list):
            method1(word_list, predictions, cuda=cuda)

        # Method 2 (see doc string)
        else:
            method2(word_list, predictions, cuda=cuda)
    return predictions

def use_method1(word_list):
    if len(word_list) < 10:
        return True

    short_hns = len([img for _, img in word_list if 1 <= len(img.char_images) <= 2])
    long_hns = len([img for _, img in word_list if len(img.char_images) > 2])
    return short_hns > long_hns

def method1(word_list, predictions, cuda=False):
    num_upside_downs = 0

    # Counts for upside down 3s, 4s, and 7s for both orientations
    for word_id, img in word_list:
        if img.char_images:     # word can be read by connected components
            for char in img.char_images:
                pred = predict_digit(char, const.UPSIDE_DOWN_MODEL, cuda=cuda)
                u_pred = predict_digit(np.rot90(char,2), const.UPSIDE_DOWN_MODEL, cuda=cuda)

                if (pred == 10) or (pred == 11) or (pred == 12):
                    num_upside_downs += 1
                if (u_pred == 10) or (u_pred == 11) or (u_pred == 12):
                    num_upside_downs -= 1

    # Case when the numbers are upside down
    if num_upside_downs > 0:
        for word_id, img in word_list:
            # Number can be segmented into individual digits
            if len(img.char_images) > 0:
                flipped_char_images = [np.rot90(digit, 2) for digit in img.char_images]
                img.char_images = flipped_char_images[::-1]
            img.word_image = np.rot90(img.word_image, 2)

    for word_id, img in word_list:
        # CODE TO USE DIGIT MODEL
        # with further testing, we found that the sequence model performed better
        # than breaking the word into individual characters and processing them
        # individually.  If the digit model, and the method of breaking up words
        # into connected components is tuned to outperform sequence model, then
        # restore this code
        # if len(img.char_images) > 0:
        #     pred_word = ""

        #     for char in img.char_images:
        #         pred = predict_digit(char, digit_model, cuda=cuda)

        #         pred_word += str(pred)
        #     predictions[word_id] = pred_word

        # # Number cannot be segmented into individual digits
        # else:
        pred_word = predict_word(img.word_image, const.NUMBER_MODEL, number_only=True, cuda=cuda)
        predictions[word_id] = pred_word


def method2(word_list, predictions, cuda = False):
    first_chars = {}
    u_first_chars = {}

    u_temp_pred = {}
    temp_pred = {}

    for word_id, img in word_list:
        # CODE TO USE INDIVIDUAL DIGIT MODEL (see note in above comment block)
        # Number can be segmented into individual digits
        # if img.char_images:
        #     pred_word = ""
        #     u_pred_word = ""

        #     for char in img.char_images:
        #         datapoint = char
        #         pred = predict_digit(datapoint, digit_model, cuda=cuda)
        #         u_pred = predict_digit(np.rot90(datapoint, 2), digit_model, cuda=cuda)
        #         pred_word += str(pred)
        #         u_pred_word = str(u_pred) + u_pred_word

        #     if pred_word[0] not in first_chars:
        #         first_chars[pred_word[0]] = 1
        #     else:
        #         first_chars[pred_word[0]] += 1

        #     if u_pred_word[0] not in u_first_chars:
        #         u_first_chars[u_pred_word[0]] = 1
        #     else:
        #         u_first_chars[u_pred_word[0]] += 1

        #     temp_pred[word_id] = pred_word
        #     u_temp_pred[word_id] = u_pred_word

        # Number cannot be segmented into individual digits
        #else:
        datapoint = img.word_image

        pred = predict_word(datapoint, const.NUMBER_MODEL, number_only=True, cuda=cuda)
        temp_pred[word_id] = pred

        u_pred = predict_word(np.rot90(datapoint, 2), const.NUMBER_MODEL, number_only=True, cuda=cuda)
        u_temp_pred[word_id] = u_pred


        if len(pred) > 0:
            if pred[0] not in first_chars:
                first_chars[pred[0]] = 1
            else:
                first_chars[pred[0]] += 1

        if len(u_pred) > 0:
            if u_pred[0] not in u_first_chars:
                u_first_chars[u_pred[0]] = 1
            else:
                u_first_chars[u_pred[0]] += 1


    # Go for lowest entropy:
    # - sum_i p_i * log_2(p_i)
    props = []
    for i in range(10):
        if str(i) in first_chars:
            props.append(first_chars[str(i)]/len(word_list))
        else:
            props.append(0)

    u_props = []
    for i in range(10):
        if str(i) in u_first_chars:
            u_props.append(u_first_chars[str(i)]/len(word_list))
        else:
            u_props.append(0)

    ent = entropy(props)
    u_ent = entropy(u_props)


    # rightside up numbers
    if ent <= u_ent:
        for key in temp_pred:
            predictions[key] = temp_pred[key]

    # upside down numbers
    else:
        for key in u_temp_pred:
            predictions[key] = u_temp_pred[key]
        for word_id, img in word_list:
            if len(img.char_images) > 0:
                flipped_char_images = [np.rot90(char, 2) for char in img.char_images]
                img.char_images = flipped_char_images[::-1]
            img.word_image = np.rot90(img.word_image, 2)

def pred_street_names(street_names, cuda=False):
    predictions = {}
    for key in street_names:
        word_list = street_names[key]
        for word_id, img in word_list:
            pred1 = predict_word(img.word_image, const.WORD_MODEL, number_only=False, cuda=cuda)

            img_flip = np.rot90(img.word_image, 2)
            pred_flip = predict_word(img_flip, const.WORD_MODEL, number_only=False, cuda=cuda)

            predictions[word_id] = pred1, pred_flip


    return predictions
