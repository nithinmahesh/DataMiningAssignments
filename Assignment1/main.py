
import sys
import arff
import pprint
import random
import math
from arff import  *

g = 0
def ChildCount():
    global g
    g += 1
    #print(str(g))
    
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', value = None, children=None):
        self.name = name
        self.value = value
        self.children = []
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

def ChooseBestAttribute(Examples, Targetattribute, Attributes):
    usegainratio = True
    if usegainratio:
        return ChooseBestAttributeByGainRatio(Examples, Targetattribute, Attributes)
    else:
        return ChooseBestAttributeByGain(Examples, Targetattribute, Attributes)
    
def ChooseBestAttributeByGain(Examples, Targetattribute, Attributes):
    pos = neg = totalCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    for data in Examples:
        if data[targetAttrIndex] == "True":
            pos += 1
            totalCount += 1
        elif data[targetAttrIndex] == "False":
            neg += 1
            totalCount += 1
    entropys = 0
    if pos != 0:
        entropys -= pos/(totalCount) * math.log2(pos/(totalCount))
    if neg != 0:
        entropys -= neg/(totalCount) * math.log2(neg/(totalCount))
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
                        if data[targetAttrIndex] == "True":
                            pos += 1
                        elif data[targetAttrIndex] == "False":
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
    print("Best gain is " + str(bestgain))
    if bestattr is None:
        CustomPrint("Returning none attr as best attr")
    return bestattr
    
def ChooseBestAttributeByGainRatio(Examples, Targetattribute, Attributes):
    pos = neg = totalCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    for data in Examples:
        if data[targetAttrIndex] == "True":
            pos += 1
            totalCount += 1
        elif data[targetAttrIndex] == "False":
            neg += 1
            totalCount += 1
    entropys = 0
    if pos != 0:
        entropys -= pos/(totalCount) * math.log2(pos/(totalCount))
    if neg != 0:
        entropys -= neg/(totalCount) * math.log2(neg/(totalCount))
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
                        if data[targetAttrIndex] == "True":
                            pos += 1
                        elif data[targetAttrIndex] == "False":
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
    print("Best gain ratio is " + str(bestgainratio))
    if bestattr is None:
        CustomPrint("Returning none attr as best attr")
    return bestattr
        
def GetAttrIndex(Attributes, TargetAttribute):
    for i in range(len(Attributes)):
        if Attributes[i][0] == TargetAttribute[0]:
            return i
    
def ID3(Examples, Targetattribute, Attributes):
    "Algorithm for ID3 Machine Learning"
    CustomPrint("TotalDataCount:" + str(len(Examples)))
    CustomPrint("TotalAttrinDataCount:" + str(len(Examples[0])))
    CustomPrint("TotalAttributeCount:" + str(len(Attributes)))
    root = Tree();
    positiveCount = negativeCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    CustomPrint("targetAttrIndex=" + str(targetAttrIndex))
    for member in Examples:
        if member[targetAttrIndex] == 'True':
            positiveCount += 1
        if member[targetAttrIndex] == 'False':
            negativeCount += 1
    CustomPrint(positiveCount)
    CustomPrint(negativeCount)
    if positiveCount+negativeCount != len(Examples):
        CustomPrint("Bad comparisons")
    if negativeCount == 0:
        root.assign_label("True")
        CustomPrint("Assigned True")
        return root
    if positiveCount == 0:
        root.assign_label("False")
        CustomPrint("Assigned False")
        return root
    if len(Attributes) == 0 or len(Attributes) == 1:
        if positiveCount > negativeCount:
            root.assign_label("True")
            CustomPrint("Assigned True")
            return root
        else:
            root.assign_label("False")
            CustomPrint("Assigned False")
            return root
        
    bestAttr = ChooseBestAttribute(Examples, Targetattribute, Attributes)
    if bestAttr is None:
        if positiveCount > negativeCount:
            root.assign_label("True")
            CustomPrint("Assigned True")
            return root
        else:
            root.assign_label("False")
            CustomPrint("Assigned False")
            return root
    bestAttrIndex = GetAttrIndex(Attributes, bestAttr)
    
    CustomPrint("Best Attr:" + bestAttr[0])
    CustomPrint("Best Attr Indx:" + str(bestAttrIndex))
    root.assign_label(bestAttr[0])
        
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
                child.assign_label("True")
            else:
                CustomPrint("Assigned False")
                child.assign_label("False")
        else:
            newAttrSet = Attributes[:]
            del newAttrSet[bestAttrIndex]
            for member in exampleSubset:
                del member[bestAttrIndex]
            child = ID3(exampleSubset, Targetattribute, newAttrSet)
            root.add_child(child)
            child.assign_value(value)
    if len(root.children) == 0:
        print("Attribute node does not have any child")
    return root
 
class Queue:
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
    
def Evaluate(tree, testSet, TargetAttribute, Attributes):
    print("At Eval total Attr:" + str(len(Attributes)))
    print("At Eval total testSet Attr:" + str(len(testSet[0])))
    totalCount = len(testSet)
    positive = good = 0
    targetAttrIndex = GetAttrIndex(Attributes, TargetAttribute)
    for test in testSet:
        expected = test[targetAttrIndex]
        actual = GetPrediction(tree, test, TargetAttribute, Attributes)
        CustomPrint(test)
        #print("Actual:" + str(actual) + " Expected:" + expected)
        if expected == actual:
            positive += 1
        if actual == "True" or actual == "False":
            good += 1
    print("Good eval:" + str(float(good/totalCount)*100))
    return float(positive/totalCount)*100

def GetPrediction(tree, test, TargetAttribute, Attributes):
    if str(tree) == "True":
        return "True"
    elif str(tree) == "False":
        return "False"
    else:
        for attr in Attributes:
            if str(tree) == attr[0]:
                for child in tree.children: 
                    if child.value == test[GetAttrIndex(Attributes, attr)]:
                        return GetPrediction(child, test, TargetAttribute, Attributes)
                print("Missed path with test[value]:" + test[GetAttrIndex(Attributes, attr)])
                for child in tree.children: 
                    print(child.value)
    print("Bad evaluation at node:" + tree.name + " " + str(tree.value))
    return str(tree)
    
def CustomPrint(string):
    #print(string)
    return string

a = arff.load(open('training_subsetD.arff'))
#a = arff.load(open(r'C:\Users\nithinm\Documents\MachineLearningPedro\Assignment1\Data\AssignmentData\test.arff'))

#pprint.pprint(a['attributes'])
#pprint.pprint(a['data'][0])

for attr in a['attributes']:
    attr[1].append(None)

tree = ID3(a['data'], a['attributes'][-1], a['attributes'])

PrintTree(tree)

print("Tree built with nodecount:" + str(g))

validationData = arff.load(open('validation_subsetD.arff'))
#validationData = arff.load(open(r'C:\Users\nithinm\Documents\MachineLearningPedro\Assignment1\Data\AssignmentData\test.arff'))

for attr in validationData['attributes']:
    attr[1].append(None)


print("Validation Accuracy:" + str(Evaluate(tree, validationData['data'], validationData['attributes'][-1], validationData['attributes'])))

testData = arff.load(open('testingD.arff'))


for attr in testData['attributes']:
    attr[1].append(None)

print("Testing Accuracy:" + str(Evaluate(tree, testData['data'], testData['attributes'][-1], testData['attributes'])))

# print ("Hello, Python!")
# x=15
# x += 1
# #x = 'foo'; 
# CustomPrint(str(x))

# list = ['abcd', 786 , 2.23, 'john', 70.2 ]
# CustomPrint(list)