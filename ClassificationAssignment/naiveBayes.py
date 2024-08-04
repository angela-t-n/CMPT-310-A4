# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#

import util
import classificationMethod
import math
import random

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        super().__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k






    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in y our code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))

        if (self.automaticTuning):
            kgrid = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.5, 1, 5, 10, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)






    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        bestAccuracyCount = -1 # best accuracy so far on validation set

        # Common training - get all counts from training data
        # We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter() # probability over labels
        commonConditionalProb = util.Counter() # Conditional probability of feature feat being 1
                                      # indexed by (feat, label)
        commonCounts = util.Counter() # how many time I have seen feature 'feat' with label 'y'
                                    # whatever inactive or active
        bestParams = (commonPrior, commonConditionalProb, kgrid[0])

        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = trainingLabels[i]
            "*** Y OUR CODE HERE to complete populating commonPrior, commonCounts, and commonConditionalProb ***"
            #util.raiseNotDefined()

            '''
            # basically referring to the lecture slide for this one
            # there's two things we need i think:
                # estimation of P(label), how often does each label occur
                # estimation of P("probability of every individual feature happening given a label existing") (P(F_i | label)), 
                    # which is basically, bayes theorem describing how the label affects f in a list of features
            
            # then, we need to calculate the classification
            # to do so, we need to evaluate all the features possible, for a label as evidence
            # we also need to query the probability of a label given all the possible features (P(label | f1, f2, ... fn))

            # putting that together we somehow get this
            # key = (label, feature_i)
            # P(key) = P(Label) some weird symbol that stands for i? P(F_i | label)
                # oh the weird symbol means chain rule
            # which can be visually expanded into this giant list on slide 22

            # that slide also breaks it down into the following steps:
                # 1. get a joint probability of the label and evidence for each label
                    # [P(y1| f1 to fn) ... P(yn| f1 to fn) 
                
                # 2. sum that all up?? which results in P(f1 to fn)???
                
                # 3. Normalize by dividing step 1 by step 2
                    # results in P(label | f1 to fn)

            # oh apparently that was all inference, as stated in slide 33.
            # we then estimate it as:
                # P(label), the "prior" over labels
                # P(F_i | label) for each feature (evidence)

                # collectively called parameters, and denoted as theta, and are typically from training data counts
                # so if theta appears, i know what to do i guess???
            '''
            
            # basically this is the part 1 portion

            # tally the common probability over the current label
            # the result of this will be the "how often each label occur" part i guess
            commonPrior[label] += 1

            # now, for every piece of data in the datum, 
            # check if the feature at that specific data is greater than 0
            # if so, tally it up into commonConditionalProb (indexed by feat, label as seen above)

            for f, val in datum.items():
                # essentially the P(F_i | Label) part
                if val > 0:
                    # tally since it has a positive datum at that feature
                    commonConditionalProb[(f, label)] += 1

                # and also tally up the fact that we visited this feature at our current label
                commonCounts[((f, label))] += 1


            
        # lapace estimate stuff from slide 43?
        # basically, pretend we saw every outcome k more times than we actually did
        # for every pixel in the grid
        # explaination of why is on slide 40~41

        # essentially, for the existing count for that feature for that label
        # add an extra k time to it

        for k in kgrid: # Smoothing parameter tuning loop!
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in commonPrior.items():
                prior[key] += val
            for key, val in commonCounts.items():
                counts[key] += val
            for key, val in commonConditionalProb.items():
                conditionalProb[key] += val




            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    "*** Y OUR CODE HERE to update conditionalProb and counts using Lablace smoothing ***"
                    #util.raiseNotDefined()

                    # just chuck an extra k times to it!
                    # NEEDS TO BE 2*k, k times for the presence and another k times for the lack of
                    # idk that's what some blog said when simplying it
                    # i guess that's also what |X| is in this case, presence and no presence?
                    counts[(feat, label)] += 2*k

                    # this one uses the lapace for conditionals on slide 44
                    # just gonna overwrite the work done earlier lol
                    #conditionalProb[(feat, label)] = (commonConditionalProb[(feat, label)] + k) / (commonCounts[(feat, label)] + 2*k)

                    # OH CUZ IT NORMALIZES IT LATER FHSDIOFSIOFIOJSD I DON'T NEED THAT
                    conditionalProb[(feat, label)] += k


            #i guess now the following code is supposed to normalize it?



            # normalizing:
            prior.normalize()
            for x, count in conditionalProb.items():
                conditionalProb[x] = count * 1.0 / counts[x]

            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation set
            predictions = self.classify(validationData)
            accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print("Performance on validation set for k=%f: (%.1f%%)" % (k, 100.0*accuracyCount/len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior, conditionalProb, k)
                bestAccuracyCount = accuracyCount
            # end of automatic tuning loop

        self.prior, self.conditionalProb, self.k = bestParams
        print("Best Performance on validation set for k=%f: (%.1f%%)" % (self.k, 100.0 * bestAccuracyCount / len(validationLabels)))







    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses









    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            "*** Y OUR CODE HERE, to populate logJoint() list ***"
            #util.raiseNotDefined()

            # idk what the hell this part is for, but I'm just practically
            # guessing from that comment above
            # and chapter 6 of the textbook...

            # essentially, for every label
            # we are estimating the log probability, and storing it in the log-joint counter
            for f, val in datum.items():
                estimateOfLog = 0

                # check our datum for negatives, since we can't take the log of a negative value
                if val > 0:
                    # positive value, so we can just use it
                    estimateOfLog = math.log(self.conditionalProb[(f,label)])
                else:
                    # negtive value, take the compliment and use that in our calculations that instead
                    estimateOfLog = math.log(1 - self.conditionalProb[(f,label)])

                # once done, store the estimate into the array
                # MUST be += because we need to accumulate it for that label
                logJoint[label] += estimateOfLog

        # returns a list of log estimates for every valid label
        return logJoint









    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** Y OUR CODE HERE, to populate featureOdds based on above formula. ***"
        #util.raiseNotDefined()

        # using the common above, we're basically just dividing the conditional probability
        # given the current feature in a list of features, and 2 seperate labels
        for f in self.features:
            probability = self.conditionalProb[(f, label1)] / self.conditionalProb[(f,label2)]

            # append this for the current feature as a pair of (feature, resulting probability)
            featuresOdds.append((f, probability))

        # then, similar to previously, sort this from highest to lowest
        # had to google this one since it's not a util thing, can't just sort keys using it's values
        featuresOdds.sort(key=lambda pair: pair[1], reverse=True)

        # return the top 100 features by removing the probabilities in the tuple pairs
        # and making a list of just the keys
        # can be done by converting the list of tuples (which is basically a dict) into a dict
        # and then just yoinking the keys
        featureList = dict(featuresOdds[:100])
        featureList = featureList.keys()

        return featureList

        #return featuresOdds
