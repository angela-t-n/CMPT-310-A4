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

        bestAccuracyCount = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()


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
