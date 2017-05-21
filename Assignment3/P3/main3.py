
import sys
import arff
import pprint
import random
import math
from arff import  *
from scipy.stats import chi2

# -------------------------------------------------------------------------------------------
# Tree Node representation
# -------------------------------------------------------------------------------------------

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', value = None, children=None):
        self.name = name
        self.value = value
        self.children = []
        self.positive = 0
        self.negative = 0
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
        ChildCount()
    def assign_label(self, label):
        self.name=label
    def assign_value(self, value):
        self.value = value
    def assign_counts(self, pos, neg):
        self.positive = pos
        self.negative = neg

# -------------------------------------------------------------------------------------------
# ID3 decision tree builder
# -------------------------------------------------------------------------------------------

def ID3(Examples, Targetattribute, Attributes, confidence, usegainratio):
    "Algorithm for ID3 Machine Learning"
    CustomPrint("TotalDataCount:" + str(len(Examples)))
    CustomPrint("TotalAttrinDataCount:" + str(len(Examples[0])))
    CustomPrint("TotalAttributeCount:" + str(len(Attributes)))
    root = Tree();
    positiveCount = negativeCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    CustomPrint("targetAttrIndex=" + str(targetAttrIndex))
    
    # Count positive and negative examples
    for member in Examples:
        if member[targetAttrIndex] == '+':
            positiveCount += 1
        if member[targetAttrIndex] == '-':
            negativeCount += 1
    CustomPrint(positiveCount)
    CustomPrint(negativeCount)
    if positiveCount+negativeCount != len(Examples):
        CustomPrint("Bad comparisons")
    root.assign_counts(positiveCount, negativeCount)
        
    # All positive case
    if negativeCount == 0:
        root.assign_label("+")
        CustomPrint("Assigned True")
        return root
        
    # All negative case
    if positiveCount == 0:
        root.assign_label("-")
        CustomPrint("Assigned False")
        return root
        
    # Out of attributes - assign the most occurring label
    if len(Attributes) == 0 or len(Attributes) == 1:
        if positiveCount > negativeCount:
            root.assign_label("+")
            CustomPrint("Assigned True")
            return root
        else:
            root.assign_label("-")
            CustomPrint("Assigned False")
            return root
        
    # Choose the best attribute based on the given setting
    bestAttr = ChooseBestAttribute(Examples, Targetattribute, Attributes, confidence, usegainratio)
    
    # If we did not find an attribute, assign the most occurring label
    if bestAttr is None:
        if positiveCount > negativeCount:
            root.assign_label("+")
            CustomPrint("Assigned True")
            return root
        else:
            root.assign_label("-")
            CustomPrint("Assigned False")
            return root
    bestAttrIndex = GetAttrIndex(Attributes, bestAttr)
    
    CustomPrint("Best Attr:" + bestAttr[0])
    CustomPrint("Best Attr Indx:" + str(bestAttrIndex))
    
    # Assign label as attribute name
    root.assign_label(bestAttr[0])
        
    # Add a branch for each possible value
    for value in bestAttr[1]:
        CustomPrint("Adding child with value:" + str(value))
        exampleSubset = []
        for member in Examples:
            #CustomPrint(member[bestAttrIndex])
            if member[bestAttrIndex] == value:
                newmember = member[:]
                exampleSubset.append(newmember)
        CustomPrint("Child has count:" + str(len(exampleSubset)))
        if len(exampleSubset) == 0:
            child = Tree('root', value, None)
            root.add_child(child)
            if positiveCount > negativeCount:
                CustomPrint("Assigned True")
                child.assign_label("+")
            else:
                CustomPrint("Assigned False")
                child.assign_label("-")
        else:
            newAttrSet = Attributes[:]
            del newAttrSet[bestAttrIndex]
            for member in exampleSubset:
                del member[bestAttrIndex]
            child = ID3(exampleSubset, Targetattribute, newAttrSet, confidence, usegainratio)
            root.add_child(child)
            child.assign_value(value)
    if len(root.children) == 0:
        print("Attribute node does not have any child")
    return root
 
# -------------------------------------------------------------------------------------------
# Choosing best attribute logic
# -------------------------------------------------------------------------------------------

def ChooseBestAttribute(Examples, Targetattribute, Attributes, confidence, usegainratio):
    "Chooses best attribute for the current node"
    bestattr = None
    
    # First get the best attribute
    if usegainratio:
        bestattr = ChooseBestAttributeByGainRatio(Examples, Targetattribute, Attributes)
    else:
        bestattr = ChooseBestAttributeByGain(Examples, Targetattribute, Attributes)
    
    return bestattr
        
    # Verify through ChiSquare test that it is indeed worth adding this branch
    # if ShouldStopByChiSquare(Examples, Targetattribute, Attributes, bestattr, confidence):
        # return None
    # else:
        # return bestattr
    
def ChooseBestAttributeByGain(Examples, Targetattribute, Attributes):
    "Choose the best attribute by the gain in entropy it provides"
    pos = neg = totalCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    
    # Count the positive and negative examples
    for data in Examples:
        if data[targetAttrIndex] == "+":
            pos += 1
            totalCount += 1
        elif data[targetAttrIndex] == "-":
            neg += 1
            totalCount += 1
            
    # Calculate entropy of the current examples
    entropys = 0
    if pos != 0:
        entropys -= pos/(totalCount) * math.log2(pos/(totalCount))
    if neg != 0:
        entropys -= neg/(totalCount) * math.log2(neg/(totalCount))
    
    # Calculate entropy of each child and assess the gain this attribute provides
    bestgain = 0
    bestattr = None
    for attr in Attributes:
        if attr != Targetattribute:
            gain = entropys
            attrIndex = GetAttrIndex(Attributes, attr)
            for value in attr[1]:
                pos = neg = count = 0
                for data in Examples:
                    if data[attrIndex] == value:
                        count += 1
                        if data[targetAttrIndex] == "+":
                            pos += 1
                        elif data[targetAttrIndex] == "-":
                            neg += 1
                if count != 0:
                    #CustomPrint(str(count) + " " + str(pos) + " " + str(neg))
                    newentropy = 0
                    if pos != 0:
                        newentropy -= pos/(count) * math.log2(pos/(count))
                    if neg != 0:
                        newentropy -= neg/(count) * math.log2(neg/(count))
                    gain -= count/totalCount * newentropy
            if gain > bestgain and gain != entropys:
                bestattr = attr
                bestgain = gain
    CustomPrint("Best gain is " + str(bestgain))
    
    # Return the best attribute which gives the highest gain
    if bestattr is None:
        CustomPrint("Returning none attr as best attr")
    return bestattr
    
def ChooseBestAttributeByGainRatio(Examples, Targetattribute, Attributes):
    "Choose best attribute by highest gain ratio"
    pos = neg = totalCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    
    # Count the positive and negative examples
    for data in Examples:
        if data[targetAttrIndex] == "+":
            pos += 1
            totalCount += 1
        elif data[targetAttrIndex] == "-":
            neg += 1
            totalCount += 1
    
    # Calculate entropy of the current examples
    entropys = 0
    if pos != 0:
        entropys -= pos/(totalCount) * math.log2(pos/(totalCount))
    if neg != 0:
        entropys -= neg/(totalCount) * math.log2(neg/(totalCount))
    
    # Calculate entropy of each child and assess the gain ratio this attribute provides
    bestgainratio = 0
    bestattr = None
    for attr in Attributes:
        if attr != Targetattribute:
            gain = entropys
            split = 0
            attrIndex = GetAttrIndex(Attributes, attr)
            for value in attr[1]:
                pos = neg = count = 0
                for data in Examples:
                    if data[attrIndex] == value:
                        count += 1
                        if data[targetAttrIndex] == "+":
                            pos += 1
                        elif data[targetAttrIndex] == "-":
                            neg += 1
                if count != 0:
                    #CustomPrint(str(count) + " " + str(pos) + " " + str(neg))
                    newentropy = 0
                    if pos != 0:
                        newentropy -= pos/(count) * math.log2(pos/(count))
                    if neg != 0:
                        newentropy -= neg/(count) * math.log2(neg/(count))
                    gain -= count/totalCount * newentropy
                    split -= count/totalCount * math.log2(count/totalCount)
            if split != 0:
                gainratio = gain/split
                if gainratio > bestgainratio and gain != entropys:
                    bestattr = attr
                    bestgainratio = gainratio
    CustomPrint("Best gain ratio is " + str(bestgainratio))
    
    # Return the best attribute which gives the highest gain
    if bestattr is None:
        CustomPrint("Returning none attr as best attr")
    return bestattr

# -------------------------------------------------------------------------------------------
# Chi Square Split Stopping
# -------------------------------------------------------------------------------------------

def ShouldStopByChiSquare(Examples, Targetattribute, Attributes, bestattr, confidence):
    "Calculates Chi Square heuristic to decide whether to stop splitting"
    if bestattr == None:
        return True
    
    # Get the corresponding critical value
    limit = chi2.isf(1 - confidence, len(bestattr[1]) - 1 - 1)
    CustomPrint("Critical value:" + str(limit))
    totalpos = totalneg = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    bestAttrIndex = GetAttrIndex(Attributes, bestattr)
    chisquare = 0
    
    # 1. Count the total positive and negative examples
    for data in Examples:
        if data[targetAttrIndex] == "+":
            totalpos += 1
        elif data[targetAttrIndex] == "-":
            totalneg += 1
            
    # Calculate chi square test statistic by iterating over every possible value
    for value in bestattr[1]:
        pos = neg = 0
        expos = exneg = 0
        for data in Examples:
            if data[bestAttrIndex] == value:
                if data[targetAttrIndex] == "+":
                    pos += 1
                elif data[targetAttrIndex] == "-":
                    neg += 1
        expos = totalpos * (pos + neg) / (totalpos + totalneg)
        exneg = totalneg * (pos + neg) / (totalpos + totalneg)
        if expos != 0:
            chisquare += (pos - expos)**2 / expos 
        if exneg != 0:
            chisquare += (neg - exneg)**2 / exneg
    CustomPrint("Chisquare:" + str(chisquare))
    
    # Reject if lesser than critical value
    if chisquare > limit:
        return False
    return True

# -------------------------------------------------------------------------------------------
# Evaluation and Prediction functions 
# -------------------------------------------------------------------------------------------

def BaggingEvaluate(treeSet, testSet, TargetAttribute, Attributes):
    "Evaluate a test set with the given built decision tree"
    CustomPrint("At Eval total Attr:" + str(len(Attributes)))
    CustomPrint("At Eval total testSet Attr:" + str(len(testSet[0])))
    totalCount = len(testSet)
    positive = good = 0
    targetAttrIndex = GetAttrIndex(Attributes, TargetAttribute)
    
    expectedPositives = expectedNegatives = predictedPositives = preditedNegatives = 0
    correctlyPredictedPositives = 0
    
    # PRedict for every test case
    for test in testSet:
        expected = test[targetAttrIndex]
        posVotes = negVotes = 0
        for i in range(len(treeSet)):
            val = GetPrediction(treeSet[i], test, TargetAttribute, Attributes)
            if val == "+":
                posVotes += 1
            else:
                negVotes += 1
        CustomPrint("Votes: " + str(posVotes) + " " + str(negVotes))
        if posVotes >= negVotes:
            actual = "+"
        else:
            actual = "-"
        if expected == "+":
            expectedPositives += 1
        if expected == "-":
            expectedNegatives += 1
        if actual == "+":
            predictedPositives += 1
        if actual == "-":
            preditedNegatives += 1
        #CustomPrint(test)
        if expected == actual:
            positive += 1
            if expected == "+":
                correctlyPredictedPositives += 1
        if actual == "+" or actual == "-":
            good += 1
    CustomPrint("Good eval:" + str(float(good/totalCount)*100))
    CustomPrint(correctlyPredictedPositives)
    CustomPrint(predictedPositives)
    CustomPrint(expectedPositives)
    
    CustomPrint("Precision: " + str(float(correctlyPredictedPositives/predictedPositives)))
    CustomPrint("Recall: " + str(float(correctlyPredictedPositives/expectedPositives)))
    
    # REturn the accuracy
    return float(positive/totalCount)*100

def Evaluate(tree, testSet, TargetAttribute, Attributes):
    "Evaluate a test set with the given built decision tree"
    CustomPrint("At Eval total Attr:" + str(len(Attributes)))
    CustomPrint("At Eval total testSet Attr:" + str(len(testSet[0])))
    totalCount = len(testSet)
    positive = good = 0
    targetAttrIndex = GetAttrIndex(Attributes, TargetAttribute)
    
    expectedPositives = expectedNegatives = predictedPositives = preditedNegatives = 0
    correctlyPredictedPositives = 0
    
    # PRedict for every test case
    for test in testSet:
        expected = test[targetAttrIndex]
        actual = GetPrediction(tree, test, TargetAttribute, Attributes)
        if expected == "+":
            expectedPositives += 1
        if expected == "-":
            expectedNegatives += 1
        if actual == "+":
            predictedPositives += 1
        if actual == "-":
            preditedNegatives += 1
        #CustomPrint(test)
        if expected == actual:
            positive += 1
            if expected == "+":
                correctlyPredictedPositives += 1
        if actual == "+" or actual == "-":
            good += 1
    CustomPrint("Good eval:" + str(float(good/totalCount)*100))
    CustomPrint(correctlyPredictedPositives)
    CustomPrint(predictedPositives)
    CustomPrint(expectedPositives)
    
    print("Precision: " + str(float(correctlyPredictedPositives/predictedPositives)))
    print("Recall: " + str(float(correctlyPredictedPositives/expectedPositives)))
    
    # REturn the accuracy
    return float(positive/totalCount)*100

def GetPrediction(tree, test, TargetAttribute, Attributes):
    "Evaluate a single test case on the given built tree and return the predicted target attribute"
    # Root is a leaf node
    if str(tree) == "+":
        return "+"
    elif str(tree) == "-":
        return "-"
    else:
        # Root is a non-leaf node. FInd the attribute it is referring to
        for attr in Attributes:
            if str(tree) == attr[0]:
                # Find the branch to continue the search by checking values on each branch for current attribute
                for child in tree.children: 
                    if child.value == test[GetAttrIndex(Attributes, attr)]:
                        # Found the branch, proceed with the prediction on this subtree
                        return GetPrediction(child, test, TargetAttribute, Attributes)
    return str(tree)

# -------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------

def ChildCount():
    "A counter to keep track of number of nodes"
    global g
    g += 1
    #print(str(g))

class Queue:
    "Generic Queue Implementation"
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def PrintTree(root):
    "Helper function to print the tree"
    myq = Queue()
    myq.enqueue(None)
    myq.enqueue(root);
    while myq.size() != 0:
        el = myq.dequeue()
        if el is None:
            CustomPrint('')
            if myq.size() != 0:
                myq.enqueue(None)
            continue
        # CustomPrint(el.value, end='')
        # CustomPrint("-", end='')
        # CustomPrint(el.name, end=' ')
        for child in el.children:
            myq.enqueue(child)

def FindAndPrintBestTrueLabel(root):
    "Find and print the best path for positive case"
    maxPositive = FindMaxValue(root, True)
    PrintBestPath(root,maxPositive, True)
    
def FindAndPrintBestFalseLabel(root):
    "Find and print the best path for negative case"
    maxPositive = FindMaxValue(root, False)
    PrintBestPath(root,maxPositive, False)
    
def PrintBestPath(curr, max, positive):
    "Prints the best path containing the max positive/negative value for the tree rooted at the given node"
    if len(curr.children) == 0:
        if positive:
            if max == curr.positive and curr.name == "+":
                print(str(curr.name) + " " + str(curr.value) + ":")
                return True
        else:
            if max == curr.negative and curr.name == "-":
                print(str(curr.name) + " " + str(curr.value) + ":")
                return True
    for child in curr.children:
        if True == PrintBestPath(child, max, positive):
            print(str(curr.name) + " " + str(curr.value) + ":")
            return True
    return False

def FindMaxValue(root, positive):
    "Find the maximum positive/negative value in the tree rooted at the given node"
    max = 0 
    myq = Queue()
    myq.enqueue(None)
    myq.enqueue(root);
    while myq.size() != 0:
        el = myq.dequeue()
        if el is None:
            CustomPrint('')
            if myq.size() != 0:
                myq.enqueue(None)
            continue
        if positive:
            if el.positive > max and len(el.children) == 0 and el.name == "+":
                max = el.positive
        else:
            if el.negative > max and len(el.children) == 0 and el.name == "-":
                max = el.negative
        for child in el.children:
            myq.enqueue(child)
    return max

def GetAttrIndex(Attributes, TargetAttribute):
    "Given an attribute and set of all attributes returns the index of the given attribute in the attribute set"
    for i in range(len(Attributes)):
        if Attributes[i][0] == TargetAttribute[0]:
            return i
    
def CustomPrint(string):
    #print(string)
    return string

    
def Sample(data):
    newdata = []
    for i in range(len(data)):
        newdata.append(data[random.randint(0,len(data) - 1)]);
    
    CustomPrint(len(newdata))
    return newdata

# -------------------------------------------------------------------------------------------
# Main function logic starts here
# -------------------------------------------------------------------------------------------

# Initialize node counter
g = 0

# Load training data
a = arff.load(open('train.arff'))

# Various possible settings
# Add and remove settings as appropriate
confidenceset = [0.99] #[0, 0.95, 0.99]
usegainratioset = [False] #[True, False]
samplingCount = [1,3,5,10,20]
runCount = 1000

# Iterate over all possible settings
for sampleCount in samplingCount:
    for confidence in confidenceset:
        for usegainratio in usegainratioset:
            sumAccuracy = 0
            for j in range(runCount):
                # Reset node counter
                g=0
                #print("Configuration Confidence:" + str(confidence) + " Usegainratio:" + str(usegainratio))
                
                treeSet = []
                for i in range(sampleCount):
                    treeSet.append(ID3(Sample(a['data']), a['attributes'][0], a['attributes'], confidence, usegainratio))

                #PrintTree(tree)
                
                #FindAndPrintBestTrueLabel(tree)
                #FindAndPrintBestFalseLabel(tree)

                #print("Tree built with nodecount:" + str(g+1))

                # Load test data
                testData = arff.load(open('test.arff'))

                # Calculate the accuracy of the tree with test data 
                sumAccuracy += BaggingEvaluate(treeSet, testData['data'], testData['attributes'][0], testData['attributes'])

            print("Testing Accuracy:" + str(sumAccuracy/runCount))

# -------------------------------------------------------------------------------------------
# Main function logic ends here
# -------------------------------------------------------------------------------------------

# Testing Accuracy:75.85625
# Testing Accuracy:79.35
# Testing Accuracy:80.340625
# Testing Accuracy:80.703125
# Testing Accuracy:82.296875