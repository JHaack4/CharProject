from os import listdir
from os.path import isfile, join
import numpy as np 
import cv2
import random

def crop_width(img, W, H):
    if img.shape[1] > W:
        x = int((img.shape[1] - W)/2)
        return img[0:H, x:x+W] 
    return img

def resize_preserve_aspect_ratio(img, W, H):
    w,h = img.shape[1],img.shape[0]
    resized = cv2.resize(img, (int(float(w)/h*H), H))
    return crop_width(resized,W,H)

def pad_image(img, W, H, center=0.5):
    """ center in [0,1] determines how the image is centered within the padding """
    if img.shape[1] >= W:
        return img
    total_pad = W - img.shape[1]
    right_pad = int(total_pad * (1-center))
    left_pad = total_pad - right_pad
    return cv2.copyMakeBorder( img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT)

def crop_pad_img(img, W, H,center=0.5):
    t1,b1,l1,r1 = tightest_indices(img)
    cropped = img[0:H, max(0,l1-2):min(img.shape[1],r1+3)]
    return pad_image(crop_width(cropped,W,H),W,H)
    

def tightest_indices(img):
    # returns indices of top,bottom,left,right most white pixel
    image_indices = img.nonzero()
    image_indices = np.array(list(zip(image_indices[0], image_indices[1])))

    top_i = image_indices[0,0]
    bottom_i = image_indices[-1,0]
    
    mins = image_indices.min(axis=0)
    left_i = mins[1]
    
    maxs = image_indices.max(axis=0)
    right_i = maxs[1]
    return (top_i,bottom_i,left_i,right_i)

def tightest_crop(img,H):
    t1,b1,l1,r1 = tightest_indices(img)
    return img[0:H, max(0,l1):min(img.shape[1],r1+1)]


def overlap_concat(img1, img2, overlap_amt):
    left = img1[:,0:len(img1[0]) - overlap_amt]
    mid = img1[:,len(img1[0])-overlap_amt : len(img1[0])] + img2[:, 0:overlap_amt]
    values, counts = np.unique(mid, return_counts=True)
    values = list(values)
    overlapping = 0
    if 254 in values:
        overlapping = counts[values.index(254)]
    right = img2[:,overlap_amt:]

    return np.concatenate((left, mid, right), axis = 1), overlapping


def get_concat(img1, img2, pixel_overlap, max_join=7, min_slide_back=1, max_slide_back=10):

    for i in range(1, min(len(img1[0]), len(img2[0]))):
        new, overlap = overlap_concat(img1, img2, i)
        if overlap >= pixel_overlap or i > max_join: # max_overlap is capped at 12
            if i == 1:
                new, _ = overlap_concat(img1, img2, 1)
            else:
                slide_back = random.randrange(min_slide_back, max_slide_back)
                overlap = max(1, i - slide_back)
                new, _ = overlap_concat(img1, img2, overlap)
            return new
    new, _ = overlap_concat(img1, img2, 1)
    return new


def generate_training_example(training_images, training_labels, character_set=None,
                 W=72, H=48, full_widths_only=False, pixel_overlap=10, min_spacing=1, max_join=9):

    target_width = W if full_widths_only else int(random.random()*W)
    target_width = max(4, target_width)

    chars_list = []
    start_list = []
    stop_list = []
    img = None

    while True:

        next_char_idx = random.randint(0, len(training_images)-1)
        cur_img = training_images[next_char_idx]
        cur_label = training_labels[next_char_idx]

        # ensure selection from our character set
        if character_set is not None and len(character_set)>0:
            while cur_label not in character_set:
                next_char_idx = random.randint(0, len(training_images)-1)
                cur_img = training_images[next_char_idx]
                cur_label = training_labels[next_char_idx]

        w = cur_img.shape[1]

        if img is None:
            phase_offset = random.randint(0,w-5)
            img = cur_img[0:H, phase_offset:w]

            chars_list.append(cur_label)
            start_list.append(round(phase_offset/w,2))
        else: 
            img = tightest_crop(get_concat(img, cur_img, pixel_overlap,
                     max_join=max_join, min_slide_back=min_spacing),H)

            chars_list.append(cur_label)
            start_list.append(0.00)

        if img.shape[1] >= target_width:
            overhang = img.shape[1]-target_width
            stop_list.append(round(1-overhang/w, 2))
            
            img = img[0:H,0:target_width]
            break
        else:
            stop_list.append(1.00)

    img = pad_image(img,W,H)
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return (img, chars_list, start_list, stop_list)



def generate_test_example(test_images, test_labels, character_set=None, min_chars=2, max_chars=6,
                 H=48, full_widths_only=False, pixel_overlap=10, min_spacing=1, max_join=9):

    chars_list = []
    img = None
    num_chars = random.randint(min_chars, max_chars)

    for _ in range(num_chars):

        next_char_idx = random.randint(0, len(test_images)-1)
        cur_img = test_images[next_char_idx]
        cur_label = test_labels[next_char_idx]

        # ensure selection from our character set
        if character_set is not None and len(character_set)>0:
            while cur_label not in character_set:
                next_char_idx = random.randint(0, len(test_images)-1)
                cur_img = test_images[next_char_idx]
                cur_label = test_labels[next_char_idx]

        w = cur_img.shape[1]
        chars_list.append(cur_label)

        if img is None:
            img = cur_img[0:H, 0:w]
        else: 
            img = tightest_crop(get_concat(img, cur_img, pixel_overlap,
                     max_join=max_join, min_slide_back=min_spacing),H)

    img = tightest_crop(pad_image(img,img.shape[1]+50,H),H)
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return (img, chars_list)
