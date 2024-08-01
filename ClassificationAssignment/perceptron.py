# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
#


# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights






    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):

                "*** Y OUR CODE HERE ***"
                #util.raiseNotDefined()

                # compute score for each label:
                # gotta turn it into a list ahfoiwahfpdfew9fow it kept crashing
                # cuz I was sending a giant util.Counter()
                # instead of a list of util.Counter()s
                guessedScores = self.classify([trainingData[i]])

                #find the most optimum label:
                # using this formula:
                # y′ = arg max_y’’ score(f, y′′)

                # guessedScores should just be a normal list
                # of the max scores for each label/data within that index
                y_prime = max(guessedScores)

                #update weight if necessary:
                # compare y' to y, if y' == y, then it's correct!
                # if it isn't, then we need to update something since we were
                # supposed to guess y, but we got y' instead.
                y = trainingLabels[i]
                f = trainingData[i]
                if y_prime != y:
                    # update the weight at that y
                    # wy = wy + f
                    self.weights[y] += f
                    # wy' = wy' - f
                    self.weights[y_prime] -= f

        print("finished training")






    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
      
        "*** Y OUR CODE HERE ***"
        #util.raiseNotDefined()

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

        # sort them from highest to lowest
        featuresWeights = self.weights[label].sortedKeys()

        # limit the list to just the first 100 labels
        featuresWeights = featuresWeights[:100]

        # should now be a list in decending order of the first 100 greatest keys
        return featuresWeights
