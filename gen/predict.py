# 2 = full
# 3 = partial


import cv2
from gentraining import *
import keras.models
import matplotlib.pyplot as plt
from collections import Counter
from editdistance import EditDistanceFinder

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

def newBatch(n):
    return np.zeros((n, H, W, 1))

def addToBatch(X, i, img, x1, x2):
    st = int((72 - (x2 - x1))/2)
    if st < 0: st = 0
    end = st + x2 - x1
    if end > W: end = W
    if x2 - x1 > W: x2 = x1 + W
    X[i,0:H,st:end,0] = np.array(img[0:H,x1:x2,0])

def predictBatch(X, model):
    return model.predict(X) 

def predictionRowToAnswerTuples(y):
    answer = [(letters[i-3], round(y[i],3)) for i in range(3,29)]
    answer = sorted(answer, key = lambda x: -x[1])
    return answer


def predict2(model2, img, x1, x2):
    X = np.zeros((1, H, W, 1))
    st = int((72 - (x2 - x1))/2)
    if st < 0: st = 0
    end = st + x2 - x1
    #print("%d %d %d %d " % (st, end, x1, x2))
    X[0,0:H,st:end,0] = np.array(img[0:H,x1:x2,0])
    answer = model2.predict(X).flatten()
    answer = [(letters[i-3], round(answer[i],3)) for i in range(3,29)]
    answer = sorted(answer, key = lambda x: -x[1])
    return answer # sorted list of (letter, score) tuples

def predict3(model3, img, x1, x2):
    X = np.zeros((1, H, W, 1))
    st = int((72 - (x2 - x1))/2)
    if st < 0: st = 0
    end = st + x2 - x1
    X[0,0:H,st:end,0] = np.array(img[0:H,x1:x2,0])
    answer = model3.predict(X).flatten()
    answer = [(letters[i-3], round(answer[i],3)) for i in range(3,29)]
    answer = sorted(answer, key = lambda x: -x[1])
    return answer # sorted list of (letter, score) tuples

def dynamic(model3, img):
    """ returns (word, confidence, starts, ends) """
    
    printInfo = False
    w = img.shape[1]
    spacingWindow = 8 # how much freedom is allowed in the spacing?
    stepSize = 8 # how much space between images that are checked?
    # we require that W/stepSize is an integer.

    # DP table holds (score, subword, charStart, charEnd) 
    # for the subimage from 0:i
    table = [(-1,"",-1,-1) for i in range(w)]

    batchSize = 0
    for x2 in range(0,w,stepSize):
        for x1 in range(max(0,x2-W+1), x2):
            if x1/stepSize != int(x1/stepSize):
                continue
            batchSize += 1
    X = newBatch(batchSize)

    batchIdx = 0
    for x2 in range(0,w,stepSize):
        for x1 in range(max(0,x2-W+1), x2):
            if x1/stepSize != int(x1/stepSize):
                continue

            addToBatch(X, batchIdx, img, x1, x2+1)
            
            batchIdx += 1

    y = predictBatch(X, model3)

    batchIdx = 0
    for x2 in range(0,w,stepSize):
        for x1 in range(max(0,x2-W+1), x2):
            if x1/stepSize != int(x1/stepSize):
                continue

            answer = predictionRowToAnswerTuples(y[batchIdx])
            batchIdx += 1

            bestLetter = str(answer[0][0])
            bestProb = answer[0][1] / sum([answer[i][1] for i in range(len(answer))])
            if printInfo: print("--pred for %d %d %s %.3f" % (x1,x2,bestLetter,bestProb))
            if bestLetter == 'I':
                bestProb *= 1.1

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
        
        if printInfo: print("%d %.3f %s" % (x2, table[x2][0], table[x2][1]))    

    answerLoc = w
    while answerLoc/stepSize != int(answerLoc/stepSize) or answerLoc >= len(table):
        answerLoc -= 1
    if printInfo: print(answerLoc)
    if printInfo: print(len(table))
    return table[answerLoc]
                


def slidingWindow(model2, img):
    """  """
    
    w = img.shape[1]
    stepSize = 8 # how much space between images that are checked?
    # we require that W/stepSize is an integer.

    # table holds individual letter probs for each slice
    numSlices = int(w/stepSize) + int(W/stepSize)
    table = [[0.0 for i in range(26)] for j in range(numSlices)]

    n = 0
    for x1 in range(-W+20,w-20):
        if x1/stepSize != int(x1/stepSize):
            continue

        n += 1
        x2 = x1 + W
        idx = int((x1+W)/stepSize)
        
        x1 = max(0,x1)
        x2 = min(w,x2)

        answer = predict2(model2, img, x1, x2)
        #bestLetter = str(answer[0][0])
        #bestProb = answer[0][1] / sum([answer[i][1] for i in range(len(answer))])
        print("--pred for %d %d" % (x1,x2))
        dictanswer = dict(answer)
        y = [dictanswer[l] for l in letters ]
        table[idx][0:26] = y

    for c in range(26):
        plt.plot(range(n), [table[i][c] for i in range(n)], label=chr(65+c))
    plt.legend()
    plt.show()

    return 0
    

e = EditDistanceFinder()
cnt = Counter()
showImages = False
numCorrect = 0
sumED = 0
numPredictions = 2

# test approaches
for i in range(numPredictions):
    img,char_list,char_starts,char_ends = generate_test_example(
                    training_images, letter_keys, character_set=character_set,
					H=48, min_chars=2, max_chars=6)
    true_word = ''.join(char_list)
    print(char_list)
    print(char_starts)
    print(char_ends)

    
    conf3, word3, pred_starts, pred_ends = dynamic(model3, img)
    print("model 3: %s %.3f" % (word3, conf3))
    print(pred_starts)
    print(pred_ends)
    print("".join(char_list))

    dist,alignment = e.align(word3, true_word)
    print("edit distance: %d" % dist)
    sumED += dist
    for a in alignment:
        cnt[a] += 1
    if dist == 0:
        numCorrect += 1
 
    

    #slidingWindow(model2, img)
    if showImages:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

for a in cnt.keys():
    print("%s: %d" % (a, cnt[a]))

print("correct: %d total: %d accuracy: %.3f avgEditDist %.3f" % (numCorrect,numPredictions,float(numCorrect)/numPredictions,float(sumED)/numPredictions))