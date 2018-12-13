# Author: Jordan Haack
# CS159 Lab 4
# This file implements weighted edit distance

import numpy as np
from collections import defaultdict, Counter
import itertools


class EditDistanceFinder:

    def __init__(self):
        # probs is a dictionary that maps two characters
        # to the a probability. probs[oc][ic] stores the probability
        # that we see observed char oc in a misspelling when
        # the intended char ic was intended. The char % represents
        # the null/empty char. Defaults to zero.
        self.probs = defaultdict(lambda: defaultdict(float))

    def ins_cost(self, c):
        """ Returns the cost of inserting this character """
        return 1 - self.probs['%'][c]

    def del_cost(self, c):
        """ Returns the cost of deleting this character """
        return 1 - self.probs[c]['%']
    
    def sub_cost(self, observed_char, intended_char):
        """ Returns the cost of replacing the observed char
            with the intended char.
        """
        if observed_char == intended_char:
            return 0
        return 1 - self.probs[observed_char][intended_char]

    def align(self, observed_word, intended_word):
        """ This function takes two words as input. It returns an 
            edit distance (float), as well as a list of character tuples
            that contains an alignment of the two words with that score
        """

        # A list of actions we can use
        INSERT = 1
        DELETE = 2
        SUBSTITUTION = 3
        actions = [INSERT, DELETE, SUBSTITUTION]

        oLen = len(observed_word)
        iLen = len(intended_word)

        # the score table holds at i,j the minimum cost of the alignment
        # subproblem where we use the first i chars of observed_word
        # and the first j chars of intended_word.
        scores = np.zeros((oLen + 1, iLen + 1))
        # the reconstruction table holds the action we chose
        reconstruct = np.zeros((oLen + 1, iLen + 1))
        
        # handle the subcases where the one of the words is empty:
        for i in range(oLen): # only deletions left to do
            scores[i+1][0] = scores[i][0] + self.del_cost(observed_word[i])
            reconstruct[i+1][0] = DELETE
        for j in range(iLen): # only insertions left to do
            scores[0][j+1] = scores[0][j] + self.ins_cost(intended_word[j])
            reconstruct[0][j+1] = INSERT

        # handle the rest of the coses, where both words are non-empty
        for i,j in itertools.product(range(oLen), range(iLen)):

            # try an insertion, deletion, and substitution
            scoreInsert = self.ins_cost(intended_word[j]) + scores[i+1][j]
            scoreDelete = self.del_cost(observed_word[i]) + scores[i][j+1]
            scoreSub = self.sub_cost(observed_word[i], intended_word[j])  \
                            + scores[i][j]
            
            # store which one was optimal, and remember the action for later
            scoreList = [scoreInsert, scoreDelete, scoreSub]
            scores[i+1][j+1] = np.min(scoreList)
            reconstruct[i+1][j+1] = actions[np.argmin(scoreList)]

        # the optimal score lies at the end of the table
        bestScore = scores[oLen][iLen]

        # reconstruct the alignment sequence. Starting at the end of the
        # table, we observe the action. Depending on the action, we move
        # to an older part of the table, until both strings are empty
        alignment = []
        while oLen > 0 or iLen > 0:
            if reconstruct[oLen][iLen] == DELETE:
                alignment.append( (observed_word[oLen-1], '%') )
                oLen -= 1
            if reconstruct[oLen][iLen] == INSERT:
                alignment.append( ('%', intended_word[iLen-1]) )
                iLen -= 1
            if reconstruct[oLen][iLen] == SUBSTITUTION:
                alignment.append((observed_word[oLen-1], intended_word[iLen-1]))
                oLen -= 1
                iLen -= 1

        alignment.reverse()
        return (bestScore, alignment)

    def show_alignment(self, alignments):
        """ This function takes a list of alignments, and prints
            them in a human readable fashon 
        """
        print('Observed Word: ' + ' '.join([o for o,i in alignments]))
        print('Intended Word: ' + ' '.join([i for o,i in alignments]))


    def train(self, filename):
        """ This function takes as input a file name, and creates
            a list of tuples containing the misspellings in the file,
            where the misspelling comes first. We assume the file is
            comma delimited.

            Then, we use an iterative method to train our insertion,
            deletion, and substitution costs. On each iteration, we
            use our align dp function to generate a list of character
            alignments. Then, we use those alignments to compute new
            costs for insert/delete/substitution. We repeat until
            convergence.
        """
        misList = [] # list of tuples (misspelling, correctword)
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                misList.append((tokens[0], tokens[1]))


        previousCharAlignments = []
        while True:
            # use our current dp algorithm to get a list of
            # alignments
            charAlignments = self.train_alignments(misList)

            # we converge once the char alignments are the same
            # for two consecutive iterations.
            if charAlignments == previousCharAlignments:
                break
            previousCharAlignments = charAlignments

            # update our cost function based on the char alignments we
            # have just learned about
            self.train_costs(charAlignments)

    def train_alignments(self, misList):
        """ This function takes a list of misspellings, and returns a
            single list containing all of the character alignments
            from all of the pairs
        """
        allCharAlignments = []

        for misspelling,correct in misList:
            _, alignment = self.align(misspelling, correct)
            allCharAlignments.extend(alignment)

        return allCharAlignments

    def train_costs(self, charAlignments):
        """ This function takes a list of character alignments as input,
            and uses them to estimate the liklihood of different types
            of errors
        """
        # get a list of intended chars
        charIntended = [i for o,i in charAlignments]

        # count instances of each alignment and intended char
        countIntended = Counter(charIntended)
        countAlignments = Counter(charAlignments)

        # reset the probs dictionary to all zeros
        self.probs = defaultdict(lambda: defaultdict(float))

        # update the probabilities based on the counts. Divide by
        # the count of the intended char to obtain a probability distribution
        for oc,ic in countAlignments.keys():
            self.probs[oc][ic] = countAlignments[(oc,ic)] / countIntended[ic]





def test(aligner, observed_word, intended_word):
    """ This function tests the align function """
    dist,alignment = aligner.align(observed_word, intended_word)
    print("The distance between '%s' and '%s' is: %.3f" 
                % (observed_word, intended_word, dist) )
    aligner.show_alignment(alignment)


def main():

    e = EditDistanceFinder()
    e.train('/data/spelling/wikipedia_misspellings.txt')

    # test cases
    test(e, '', '')
    test(e, '', 'aaabb')
    test(e, 'aabbd', '')
    test(e, 'abb', 'abb')
    test(e, 'abcd', 'defg')
    test(e, 'abcde', 'cdefg')
    test(e, 'dggi', 'doggo')
    test(e, 'caugt', 'caught')
    test(e, 'cought', 'caught')
    test(e, 'aboutit', 'about it')
    test(e, 'abreviation', 'abbreviation')
    test(e, 'foucs', 'focus')
    test(e, 'ghandi', 'gandhi')
    test(e, 'gogin', 'going')
    test(e, 'incompatiblities', 'incompatibilities')
    test(e, 'qtuie', 'quiet')

    # obtain sorted list of characters
    charList = list(e.probs.keys()) + list(e.probs['%'].keys())
    charList = list(set(charList)) # make unique
    charList.sort()

    # look at insertion and deletion probabilities
    print('Insertion/Deletion Costs')
    for c in charList:
        print('%s:\t%.3f%%\t%.3f%%' 
                    % (c, 100*e.probs['%'][c], 100*e.probs[c]['%']))

    # look at 'close keys'
    # This is a list of each pair of chars that share a border on my keyboard
    closeCharList = """qa qw wa ws we es ed er rd rf rt tf tg ty yg yh yu 
         uh uj ui ij ik io ok ol op az as sz sx sd dx dc df fc fv fg gv 
         gb gh hb hn hj jn jm jk km kl zx xc cv vb bn nm """ 
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    print("First  column: Probability of key  being close on a keyboard")
    print("Second column: Probability of typo being close on a keyboard")

    for ic in alphabet:

        typoMass = 0.0
        closeTypoMass = 0.0
        numPossiblePairs = 0
        numClosePairs = 0

        for oc in alphabet:
            if oc==ic:
                continue # ignore correct spellings
            
            # update global mass/count
            typoMass += e.probs[oc][ic]
            numPossiblePairs += 1
            
            # check if the two letters are close on my keyboard
            if oc+ic in closeCharList or ic+oc in closeCharList:
                closeTypoMass += e.probs[oc][ic]
                numClosePairs += 1
    
        # print out the probability of a key/typo being close to this letter
        print('%s:\t%.3f%%\t%.3f%%' % (ic,                                     \
          (100*numClosePairs/numPossiblePairs), (100*closeTypoMass/typoMass)))
        


if __name__ == '__main__':
    main()