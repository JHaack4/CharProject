# 2 = full
# 3 = partial


import cv2
from gentraining import *
import keras.models

model2 = keras.models.load_model('../model/sanborn_rotnet2.hdf5')
model3 = keras.models.load_model('../model/sanborn_rotnet3.hdf5')
print('models loaded')

H = 48
W = 72
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
test_letter_set = 'ABCDEFGHIKLMNOPRSTUVW'
character_set = set([i for i in test_letter_set])

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

"""
# look at individual training examples
for i in range(10):
    img,char_list,start_list,stop_list = generate_training_example(training_images, 
                    letter_keys, character_set=None, full_widths_only=False,
                    W=72, H=48)
    print(char_list)
    print(start_list)
    print(stop_list)

    X = np.zeros((1, H, W, 1))
    X[0,:,:,0] = np.array(img[:,:,0])
    answer = model3.predict(X).flatten()
    answer = [(letters[i-3], round(answer[i],3)) for i in range(3,27)]
    answer = sorted(answer, key = lambda x: -x[1])
    print(answer)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

def predict2(model2, img, x1):
    X = np.zeros((1, H, W, 1))
    X[0,:,:,0] = np.array(img[:,x1:x1+W,0])
    answer = model2.predict(X).flatten()
    answer = [(letters[i-3], round(answer[i],3)) for i in range(3,27)]
    answer = sorted(answer, key = lambda x: -x[1])
    return answer # sorted list of (letter, score) tuples

def predict3(model3, img, x1, x2):
    X = np.zeros((1, H, W, 1))
    img = pad_image(tightest_crop(img[:,x1:x2],H), W, H)
    X[0,:,:,0] = np.array(img[:,:,0])
    answer = model3.predict(X).flatten()
    answer = [(letters[i-3], round(answer[i],3)) for i in range(3,27)]
    answer = sorted(answer, key = lambda x: -x[1])
    return answer # sorted list of (letter, score) tuples

def dynamic(model3, img):
    """ returns (word, confidence, starts, ends) """
    
    w = img.shape[1]
    spacingWindow = 8 # how much freedom is allowed in the spacing?
    stepSize = 8 # how much space between images that are checked?
    # we require that W/stepSize is an integer.

    # DP table holds (score, subword, charStart, charEnd) 
    # for the subimage from 0:i
    table = [(-1,"",-1,-1) for i in range(w)]

    for x2 in range(0,w,stepSize):
        for x1 in range(max(0,x2-W+1), x2):
            if x1/stepSize != int(x1/stepSize):
                continue

            answer = predict3(model3, img, x1, x2+1)
            bestLetter = str(answer[0][0])
            bestProb = answer[0][1] / sum([answer[i][1] for i in range(len(answer))])
            print("--pred for %d %d %s %.3f" % (x1,x2,bestLetter,bestProb))

            if x2-x1 > 48 and bestLetter not in 'MW':
                continue
            if x2-x1 > 24 and bestLetter in 'I':
                continue

            bestPrevString = ""
            bestPrevProb = -1
            bestPrevStarts = []
            bestPrevEnds = []

            for xp in range(max(0, x1-spacingWindow), min(x2, x1+spacingWindow)):
                if xp/stepSize != int(xp/stepSize):
                    continue # ensure evenly divisible by step size

                prevProb, prevString, prevStarts, prevEnds = table[xp]
                if prevProb > bestPrevProb:
                    bestPrevProb = prevProb
                    bestPrevString = prevString
                    bestPrevStarts = prevStarts
                    bestPrevEnds = prevEnds

            if x1 == 0:
                bestPrevString = ""
                bestPrevProb = 1
                bestPrevStarts = []
                bestPrevEnds = []

            if bestProb * bestPrevProb > table[x2][0]:
                table[x2] = (bestProb*bestPrevProb, bestPrevString+bestLetter, 
                            bestPrevStarts + [x1], bestPrevEnds + [x2])
        
        print("%d %.3f %s" % (x2, table[x2][0], table[x2][1]))    

    answerLoc = w
    while answerLoc/stepSize != int(answerLoc/stepSize) or answerLoc >= len(table):
        answerLoc -= 1
    print(answerLoc)
    print(len(table))
    return table[answerLoc]
                


def slidingWindow(model2, img):
    """  """
    pass
    


# test approaches
for i in range(10):
    img,char_list,char_starts,char_ends = generate_test_example(
                    training_images, letter_keys, character_set=character_set,
					H=48, min_chars=2, max_chars=6)
    print(char_list)
    print(char_starts)
    print(char_ends)

    conf3, word3, pred_starts, pred_ends = dynamic(model3, img)
    print("model 3: %s %.3f" % (word3, conf3))
    print(pred_starts)
    print(pred_ends)
    print("".join(char_list))

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()