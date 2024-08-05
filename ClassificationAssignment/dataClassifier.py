# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features







def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    # should calculate the number of connected white regions to determine potential shapes
        # for example, 2, 1, 3, 5, 7, have all the white regions ideally connected, or maybe 2 connected white regions if it overlaps
        # 4, 6, 9, 0 would have 2 regions at min cuz there's kinda closed regions in the numbers, and a big white region surrounding it
        # 8 would have 3, big region outside, 2 smaller regions inside. 0 could as well if theres a line across it
   
    # Copium calculates symmetry...
        # 1, 3, 8, 0 are symmetrical across the horizon
        # 1, 0, 8, is symmetrical across a vertical axis thingie idk....
        # 2, 4, 5, 6, 9, aren't symmetrical

    # the count of the amount of pixels in a quarterly region
        # 0, 1, 8 would generally have the same amount of pixels if u were to split the grid into 4 and look at it
        # 3, 4 (maybe) would be fatter on the left side
        # 2, 9, 7 are top heavy
        # 5, 6 is bottom heavy

    # the count of pixels per row and column? 
        # the way i thought about this was just... idk. disecting the entire thing
        # cuz some of them have patterns where like 1 is just kinda like a sausage, sliced up
        # 8 have gaps in the middle if sliced
        # etc...
    
    # maybe another silly one would be to test the strength of each pixel to it's neighbours (top and left neighbour only)?
    # 0 0 0
    # 1 2 1
    # 0 1 0
    # 2 would be chonky score cuz it's bigger
    # 1 would have a slightly less score cuz it's bigger than 1 but not the 0's to the left of it
    # inspiration? minecraft light levels LOL

    """
    features = basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE to extract and add enhanced features to features list ***"
    #util.raiseNotDefined()

    # firstly, the white regions:    
    # need to store the pixels we've visited in a set
    visited = set()
    numOfRegions = 0
    pixels = datum.getPixels()
    #print("pixels are: ", pixels)

    # for printing
    DEBUGPRINT = False
    
    if DEBUGPRINT:
        print("\n")

    # iterate through each pixel D:
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            # check to see if we've visited this pixel yet
            # and if not and it is lit up (means that there could be a line or shape here), 
            # then check it's neighbours to determine edges kind of thing
            # something something edge detection
            if pixels[x][y] > 0 and (x,y) not in visited:
                # increment regions cuz this black pixel is probs the start of a line
                # i kept breaking this part ahfojopwjfd
                numOfRegions += 1
            
                # create a stack of neighbours and iterate through them
                # it should eventually go through an entire area of black pixels until it hits an entirely white section
                # and thus, stops populating the stack
                # this essentially checks if this entire region is potentially together, cut off, etc
                neighbourStack = [(x,y)]
                while neighbourStack:
                    # extract then current pixel we are looking at
                    currentPixel = neighbourStack.pop()

                    # can just skip this pixel if it's already been visited
                    # maybe it'll make it faster
                    if currentPixel in visited:
                        continue
                    
                    # else: hasn't been seen.
                    # add it into visited
                    # since it is a set, it'll automatically remove dupes
                    visited.add(currentPixel)

                    # go through it's neighbours
                    #                (x, y + 1)
                    #   (x - 1, y)     (x, y)     (x + 1, y)
                    #                (x, y - 1)
                    for offx, offy in [(0, 1), (-1, 0), (1, 0), (0, -1)]:
                        neighbourx = currentPixel[0] + offx
                        neighboury = currentPixel[1] + offy

                        # check that it is within bounds
                        if(-1 < neighbourx < DIGIT_DATUM_WIDTH and -1 < neighboury < DIGIT_DATUM_HEIGHT):
                            # valid, not out of bounds pixel! check if this pixel is either white or black:
                            if pixels[x][y] > 0 and (x,y) not in visited:
                                # not a pixel we've seen yet, so add it into neighbours we need to visited to determine it's edge
                                neighbourStack.append((neighbourx, neighboury))

    # once done, we should have counted a number of regions
    # add that to the features
    # using a key that probs doesn't exist in features to add it into the dictionary thing
    # do the 3 possibility thing

    # nvm this is for some reason a god damn huge score...??
    if DEBUGPRINT:
        print("Resulting regions for this datum:", numOfRegions)
    
    features["regionScore"] = numOfRegions
    


    # next one calculates the symmetry based on the stuff above
    # similarly, we kinda have to go through the edges
    # but i guess we can do a thing where we "fold" it in half?
    # kinda like in elementary school and symmetric paintings
    # fold it in half and check if each pixels in that area light up together~
    horizontalSymScore = 0
    verticalSymScore = 0

    # horizontal first
    for x in range (DIGIT_DATUM_WIDTH):
        # go through half of the pixels
        for y in range(DIGIT_DATUM_HEIGHT // 2):
            # check if this pixel is the same as the pixel mirrored
            # (x, y)
            # -------
            # (x, height -1 (need to offset since it's stored in an array and offsetted by 1) - y)
            if pixels[x][y] == pixels[x][DIGIT_DATUM_HEIGHT - 1 - y]:
                # tally up the score
                horizontalSymScore += 1
    
    # vertical now
    # go through half of the pixels
    for x in range (DIGIT_DATUM_WIDTH // 2):
        for y in range(DIGIT_DATUM_HEIGHT):
            # check if this pixel is the same as the pixel mirrored
            # (x, y) | (width - 1 - x, y)

            if pixels[x][y] == pixels[DIGIT_DATUM_WIDTH - 1 - x][y]:
                # tally up the score
                verticalSymScore += 1
    


    # chuck it in the feature dictionary as well
    # using the same suggestion as before
    if horizontalSymScore > 0:
        features["horizontally symmetric"] = 1
    else:
        features["horizontally symmetric"] = 0
    
    if verticalSymScore > 0:
        features["vertically symmetric"] = 1
    else:
        features["vertically symmetric"] = 0

    if DEBUGPRINT:
        print("Symmetry scores: H:", horizontalSymScore, "V: ", verticalSymScore)


    # similar idea to the paper folding, kindergarten stuff
    # splitting it into quadrants and counting the pixels there would be fun too
    quadCounts = [0, 0, 0, 0]

    # go through the entire image
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            # check if this pixel is on
            if datum.getPixel(x, y) > 0:

                # if this pixel is on, now we gotta figure out the quadrants!
                # this can be done with some math, and by comparing which section it is in
                # (0 to half of the width) | (half of the width to width)

                # (0 to half of the height)
                # -----------------------------
                # (half of the height to height)

                # put these together to form the quadrants
                # top left (0 to half of the width, 0 to half of the height)
                if x < DIGIT_DATUM_WIDTH // 2 and y < DIGIT_DATUM_HEIGHT // 2:
                    quadCounts[0] += 1

                # top right (half of the width)
                elif x >= DIGIT_DATUM_WIDTH // 2 and y < DIGIT_DATUM_HEIGHT // 2:
                    quadCounts[1] += 1

                # bottom left (same deal)
                elif x < DIGIT_DATUM_WIDTH // 2 and y >= DIGIT_DATUM_HEIGHT // 2:
                    quadCounts[2] += 1
                
                # bottom right
                else:
                    quadCounts[3] += 1

    # chuck the scores into our thingie
    for i in range(4):
        quadLabel = "quadrant " + str(i)
        features[quadLabel] = quadCounts[i]

        if DEBUGPRINT:
            print("Quadrant", i, "score is: ", quadCounts[i])



    # now to slice it layer by layer?
    # column layer first
    countPixelPerCol = {}
    for x in range(DIGIT_DATUM_WIDTH):
        # initiate the dictionary for this value
        countPixelPerCol[x] = 0
        for y in range(DIGIT_DATUM_HEIGHT):
            # if the pixel exists, tally it up
            if pixels[x][y] > 0:
                countPixelPerCol[x] += 1
    
    # now row, same code but just opposite
    countPixelPerRow = {}
    for y in range(DIGIT_DATUM_HEIGHT):
        # initiate the dictionary for this value
        countPixelPerRow[y] = 0
        for x in range(DIGIT_DATUM_WIDTH):
            # if the pixel exists, tally it up
            if pixels[x][y] > 0:
                countPixelPerRow[y] += 1

    # put these individual scores into the feature as well!
    colSum = 0
    for x in range(DIGIT_DATUM_WIDTH):
        colLabel = "Column " + str(x)
        features[colLabel] = countPixelPerCol[x]
        colSum += countPixelPerCol[x]

    rowSum = 0
    for y in range(DIGIT_DATUM_HEIGHT):
        rowLabel = "Row " + str(y)
        features[rowLabel] = countPixelPerRow[y]
        rowSum += countPixelPerCol[y]


    # maybe another silly one would be to test the strength of each pixel to it's neighbours (top and left neighbour only)?
    # for example:
    # 0
    # 1
    # 2
    # 0

    # the 2 would have a higher score as it is greater than the ones around it
    # and because it's the highest values (from when i printed out the entire datum thing)
    # meanwhile 0 is lower
    for x in range(1, DIGIT_DATUM_WIDTH):
        for y in range(1, DIGIT_DATUM_HEIGHT):
            pixelLabel = f"currentPix ({x}, {y})"
            features[pixelLabel] = 0

            # if this pixel is greater than the pixels directly beside of it
            leftx = x - 1
            if leftx > 0:
                if pixels[x][y] > pixels[leftx][y]:
                    features[pixelLabel] += 1
                    

            # if this pixel is greater than the pixels directly on top of it
            topy = y - 1
            if topy > 0:
                if pixels[x][y] > pixels[x][topy]:
                    features[pixelLabel] += 1

            # adding bottom and right broke my code so i got rid of it. top and left are good enough i guess!
            # since afterall the grid i from like
            # top left corner to bottom right corner
            # on a screen so ya
            
    

    return features





def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(rawTestData[i])
            break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print (image)

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples, using smoothing 
                  factor k=2.5 and uses enhancedFeatureExtractorDigit() to extract new features 
                  for digit datum, and add them to the feature set, it also uses 
                 """

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print ("Doing classification")
    print ("--------------------")
    print ("data:\t\t" + options.data)
    print ("classifier:\t\t" + options.classifier)
    print ("using enhanced features?:\t" + str(options.features))
    print ("training set size:\t" + str(options.training))
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
    else:
        print("Unknown dataset"), options.data
        print(USAGE_STRING)
        sys.exit(2)

    legalLabels = range(10)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print ("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print (USAGE_STRING)
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print ("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print ("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
    elif(options.classifier == "perceptron"):
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print ("using automatic tuning for MIRA")
            classifier.automaticTuning = True
        else:
            print ("using default C=0.001 for MIRA")
    else:
        print ("Unknown classifier:", options.classifier)
        print (USAGE_STRING)


        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options


# Main harness code

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        printImage(features_odds)

    if(options.weights and options.classifier == "perceptron"):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

        from answers import q2
        print("Question 2's answer: ", q2())

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
