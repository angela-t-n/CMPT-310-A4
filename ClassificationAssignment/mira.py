# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)





    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # It seems to kinda start off similarish to perceptron where it applies the formula:
        # y′ = arg max_y’’ score(f, y′′)
        # but instead of scoring it up, they're just pairs of (f, y′′)?
        # and then find the max y′′

        # then, we compare this y′ to y, similar to the previous one
        # but instead of just adding or subtracting f from our weight vectors,
        # we instead need to take it by a variable step size of τ
        # This gives us the formula: 
        # wy = wy + τf
        # wy' = wy' - τf

        # τ in this case is >= 0
        # and should be a value that minimizes the distance between the weight for label c in cgrid of:
        # min_w'(0.5 * ∑_c ||{ w'_c  -  w_c}||2_2)
        # (I couldn't copypaste the formula in here...)

        # and then just whatever the rest of the PDF is saying
        # τ must also be capped by the minimum of whatever the
        # previous formula was, and a positive constant of C

        # prof provided
        bestAccuracyCount = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]

        for currentCVal in cGrid:
            # firstly, we need to 0 out the self.weights
            weights = {}
            for label in self.legalLabels:
                weights[label] = util.Counter()

            # once we have empty weights, we can then proceed to 
            # iterate through as many iterations as needed per cGrid?
            for _ in range(self.max_iterations):
                # and then for every iteration, we need to go through the training data
                # similar in perceptron
                for _ in range(len(trainingData)):
                    # for each piece of training data, we need to find the feature for it
                    f = trainingData[i]
                    y = trainingLabels[i]

                    # calculate y' using the sameish technique as perceptron
                    # classify() should return a list of scores
                    guessedScore = self.classify([trainingData[i]])
                    # get the max score to determine y'
                    y_prime = max(guessedScore)

                    # check if y' != y
                    if y_prime != y:
                        # if that's the case, now we actually need to calculate backwards r
                        # τ = min(C, ( ((w_y' - w_y) * f + 1) / (2 * f^2) )

                        weightYPrime = weights[y_prime]
                        weightY = weights[y]
                        
                        numerator = ((weightYPrime - weightY)) * f + 1.0
                        denom = (2.0 * (f * f))

                        eq = numerator / denom

                        backwards_r = min(currentCVal, eq)

                        # now we can update the weights
                        weights[y] += backwards_r * f
                        weights[y_prime] -= backwards_r * f

            # once the iteration is done, we can update self.weight
            self.weights = weights

            # now that the iteration is done, we can then compare it with some predictions
            # made in the validationData set

            # "for each C and choose the C with the highest validation accuracy"
            # "Evaluate accuracy, on the held-out validation set"
            # "In case of ties, prefer the lowest value of C"

            predictionFromValidation = self.classify(validationData)
            







        print("finished training. Best cGrid param = ", bestParams)





    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        "*** Y OUR CODE HERE ***"
        #util.raiseNotDefined()

        # tbh, probs the exact same anyway

        # classify performs this formula for every piece of data
        # and for every legal label
        # score(f, y) = ∑_i {f_i * wy_i}
        for current_f in data:
            score = util.Counter()

            # get the current score for every label
            for label in self.legalLabels:
                score[label] = current_f * self.weights[label]

            # append the max score for the current datum to our guess
            guesses.append(score.argMax())
            
        # should return a list of max scores for every iteration in data
        return guesses





    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** Y OUR CODE HERE ***"
        #util.raiseNotDefined()

        # As said in the assignment, I can just copypaste it here.
        # sort them from highest to lowest
        featuresWeights = self.weights[label].sortedKeys()

        # limit the list to just the first 100 labels
        featuresWeights = featuresWeights[:100]

        # should now be a list in decending order of the first 100 greatest keys
        return featuresWeights
