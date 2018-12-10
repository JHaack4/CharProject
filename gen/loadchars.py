
import cv2
from gentraining import *

H = 48
W = 72
character_set = set(['A','T','O','X'])

# load data
letter_path = "../data/letter/"
letter_files = [f for f in listdir(letter_path) if isfile(join(letter_path, f))]
letter_keys = [f[len(f)-5] for f in letter_files]

training_images = []
for f in letter_files:
    img = crop_width(tightest_crop(pad_image(resize_preserve_aspect_ratio(cv2.imread(letter_path + f),W,H),W,H),H),W,H)
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    training_images.append(img)
# todo: split into train/test letters
print("there are %d traning examples" % len(training_images))



for i in range(10):
    img,char_list,start_list,stop_list = generate_training_example(training_images, 
					letter_keys, character_set=character_set, full_widths_only=True,
					W=72, H=48)
    print(char_list)
    print(start_list)
    print(stop_list)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in range(10):
    img,char_list,start_list,stop_list = generate_test_example(training_images, 
					letter_keys, character_set=character_set,
					H=48, min_chars=2, max_chars=6)
    print(char_list)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()