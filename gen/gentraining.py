from os import listdir
from os.path import isfile, join
import numpy as np 
import cv2
import random

H = 48 # fixed height
W = 72 # fixed width

def crop_width(img):
    if img.shape[1] > W:
        x = int((img.shape[1] - W)/2)
        return img[0:H, x:x+W] 
    return img

def resize_preserve_aspect_ratio(img):
    w,h = img.shape[1],img.shape[0]
    resized = cv2.resize(img, (int(float(w)/h*H), H))
    return crop_width(resized)

def pad_image(img, center=0.5):
    """ center in [0,1] determines how the image is centered within the padding """
    if img.shape[1] >= W:
        return img
    total_pad = W - img.shape[1]
    right_pad = int(total_pad * (1-center))
    left_pad = total_pad - right_pad
    return cv2.copyMakeBorder( img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT)

def crop_pad_img(img, center=0.5):
    t1,b1,l1,r1 = tightest_indices(img)
    cropped = img[0:H, max(0,l1-2):min(img.shape[1],r1+3)]
    return pad_image(crop_width(cropped))
    

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

def tightest_crop(img):
    t1,b1,l1,r1 = tightest_indices(img)
    return img[0:H, max(0,l1-2):min(img.shape[1],r1+3)]


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


def get_concat(img1, img2, max_over):

    for i in range(1, min(len(img1[0]), len(img2[0]))):
        new, overlap = overlap_concat(img1, img2, i)
        if overlap >= max_over:
            if i == 1:
                new, _ = overlap_concat(img1, img2, i)
            else:
                slide_back = random.randrange(1, 10)
                overlap = max(1, i - slide_back)
                new, _ = overlap_concat(img1, img2, overlap)
            return new
    new, _ = overlap_concat(img1, img2, 1)
    return new


# load data
letter_path = "../data/letter/"
letter_files = [f for f in listdir(letter_path) if isfile(join(letter_path, f))]
letter_keys = [f[len(f)-5] for f in letter_files]

training_images = []
for f in letter_files:
    img = crop_width(tightest_crop(pad_image(resize_preserve_aspect_ratio(cv2.imread(letter_path + f)))))
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    training_images.append(img)
# todo: split into train/test letters
print("there are %d traning examples" % len(training_images))


def generate_training_example(training_images, training_labels, full_widths_only=False, max_overlap=20):
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

        w = cur_img.shape[1]

        if img is None:
            phase_offset = random.randint(0,w-5)
            img = cur_img[0:H, phase_offset:w]

            chars_list.append(cur_label)
            start_list.append(round(phase_offset/w,2))
        else: 
            img = tightest_crop(get_concat(img, cur_img, max_overlap))

            chars_list.append(cur_label)
            start_list.append(0.00)

        if img.shape[1] >= target_width:
            overhang = img.shape[1]-target_width
            stop_list.append(round(1-overhang/w, 2))
            
            img = img[0:H,0:target_width]
            break
        else:
            stop_list.append(1.00)

    img = pad_image(img)
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return (img, chars_list, start_list, stop_list)

for i in range(10):
    img,char_list,start_list,stop_list = generate_training_example(training_images, letter_keys, True)
    print(char_list)
    print(start_list)
    print(stop_list)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
""" 
img = cv2.imread(letter_path + letter_files[1])
img1 = cv2.imread(letter_path + letter_files[10])
resized_image = resize_preserve_aspect_ratio(img)
padded_image = pad_image(resized_image)
padded_image1 = pad_image(resize_preserve_aspect_ratio(img1))

print(np.shape(padded_image))
print(np.shape(padded_image1))

i = get_concat(padded_image, padded_image1, 10)
cv2.imshow('image',i)
cv2.waitKey(0)
cv2.destroyAllWindows() """