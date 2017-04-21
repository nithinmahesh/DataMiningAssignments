
import sys
import arff
import pprint
import random
from arff import  *

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', value = None, children=None):
        self.name = name
        self.value = None
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
    def assign_label(self, label):
        self.name=label
    def assign_value(self, value):
        self.value = value

def Positive():
    return True
    
def Negative():
    return False

def ChooseBestAttribute(Examples, Targetattribute, Attributes):
    return Attributes[random.randint(0, len(Attributes) - 2)]
    
def GetAttrIndex(Attributes, TargetAttribute):
    for i in range(len(Attributes)):
        if Attributes[i][0] == TargetAttribute[0]:
            return i
    
def ID3(Examples, Targetattribute, Attributes):
    "Algorithm for ID3 Machine Learning"
    print("TotalDataCount:" + str(len(Examples)))
    print("TotalAttrinDataCount:" + str(len(Examples[0])))
    print("TotalAttributeCount:" + str(len(Attributes)))
    root = Tree();
    positiveCount = negativeCount = 0
    targetAttrIndex = GetAttrIndex(Attributes, Targetattribute)
    print("targetAttrIndex=" + str(targetAttrIndex))
    for member in Examples:
        if member[targetAttrIndex] == 'True':
            positiveCount += 1
        if member[targetAttrIndex] == 'False':
            negativeCount += 1
    print(positiveCount)
    print(negativeCount)
    if positiveCount+negativeCount != len(Examples):
        print("Bad comparisons")
    if negativeCount == 0:
        root.assign_label("Positive")
        print("Assigned Positive")
        return root
    if positiveCount == 0:
        root.assign_label("Negative")
        print("Assigned Negative")
        return root
    if len(Attributes) == 0:
        if positiveCount > negativeCount:
            root.assign_label("Positive")
            print("Assigned Positive")
            return root
        else:
            root.assign_label("Negative")
            print("Assigned Negative")
            return root
        
    bestAttr = ChooseBestAttribute(Examples, Targetattribute, Attributes)
    bestAttrIndex = GetAttrIndex(Attributes, bestAttr)
    
    print("Best Attr:" + bestAttr[0])
    print("Best Attr Indx:" + str(bestAttrIndex))
    root.assign_label(bestAttr[0])
    
    for value in bestAttr[1]:
        child = Tree('root', value, None)
        root.add_child(child)
        print("Adding child with value:" + value)
        exampleSubset = []
        for member in Examples:
            #print(member[bestAttrIndex])
            if member[bestAttrIndex] == value:
                newmember = member[:]
                exampleSubset.append(newmember)
        print("Child has count:" + str(len(exampleSubset)))
        if len(exampleSubset) == 0:
            if positiveCount > negativeCount:
                print("Assigned Positive")
                child.assign_label("Positive")
            else:
                print("Assigned Negative")
                child.assign_label("Negative")
        else:
            newAttrSet = Attributes[:]
            del newAttrSet[bestAttrIndex]
            for member in exampleSubset:
                del member[bestAttrIndex]
            child.add_child(ID3(exampleSubset, Targetattribute, newAttrSet))
    return root
 
def PrintTree(root):
    print(root.name)
    print(root.value)
    for child in root.children:
        PrintTree(child)

 
a = arff.load(open('training_subsetD.arff'))
#a = arff.load(open(r'C:\Users\nithinm\Documents\MachineLearningPedro\Assignment1\Data\AssignmentData\test.arff'))

#pprint.pprint(a['attributes'])
#pprint.pprint(a['data'][0])

root = ID3(a['data'], a['attributes'][-1], a['attributes'])

#PrintTree(root)





# print ("Hello, Python!")
# x=15
# x += 1
# #x = 'foo'; 
# print(str(x))

# list = ['abcd', 786 , 2.23, 'john', 70.2 ]
# print(list)