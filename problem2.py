import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools, random, math
from operator import attrgetter

df = pd.read_csv('./arrhythmia.csv', header=None, na_values="?")

for i in range(280):
    if df[i].isnull().sum() > 0:
        df.loc[:,i].fillna(df[i].mode()[0], inplace=True)

class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.label = None
        self.data = None
        self.split = None
        self.children = []
        
    def __repr__(self):
        data = self.data
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {data[data.columns[0]].count()} rows; split" 
                f" {self.split.split_column} at {self.split.point:.2f} for children with" 
                f" {[p[p.columns[0]].count() for p in self.split.partitions()]} rows"
                f" and infomation gain {self.split.info_gain:.5f}")
        else:
            s = (f"{self.name} Leaf with {data[data.columns[0]].count()} rows, and label"
                 f" {self.label}")
        return s
    def treeTraverse(self,test):
        if self.node_type=='leaf':
            print("rows: "+str(list(test.index))+ "  will have value " + str(self.label))
        else:
            nodeTest=self.split
            splitCol=nodeTest.split_column
            splitPoint=nodeTest.point
            ltSplit=test[test[splitCol]<splitPoint]
            gtSplit=test[test[splitCol]>=splitPoint]
            if not ltSplit.empty>0:
                self.children[0].treeTraverse(ltSplit)
            if not gtSplit.empty>0:
                self.children[1].treeTraverse(gtSplit)
            
              
class Split(object):
    def __init__(self, data, class_column, split_column, point=None, subset=None):  #otherwise return the attribute

        self.data = data
        self.class_column = class_column
        self.split_column = split_column
        self.info_gain = None
        self.point = point
        self.partition_list = None # stores the data points on each side of the split
        self.compute_info_gain()
        
    def compute_info_gain(self):
        parentCount=self.data[0].count()
        parentOutcomes=self.data.loc[:,self.class_column].value_counts().apply(lambda x:x/parentCount)
        parentEntropy=parentOutcomes.apply(lambda x:-x*math.log(x,2)).sum()

        attrVals=set(self.data[self.split_column])
        if attrVals=={0,1}:
            thisEntropy=[]
            data=self.data
            curMin=[0,100000]
            for thisVal in attrVals:
                thisValOutcomes=data.loc[data[self.split_column]==thisVal] #dataframe of all rows with this value in this attribute
                countVal=thisValOutcomes[0].count()
                outcomeCounts=thisValOutcomes[self.class_column].value_counts()
                outcomeProportions=outcomeCounts.apply(lambda x:x/countVal) #get proportions of each outcome with this self.split_column value
                entropy=outcomeProportions.apply(lambda x:-x*math.log(x,2)).sum()

                if entropy<curMin[1]:
                    curMin=[thisVal,entropy]
            cutVal=curMin
        else:
            data=self.data
            entropy=[]
            splits=[] #list of potential split points
            attrVals=sorted(list(attrVals)) #convert set of attribute values to list and sort
            numPoints=data[0].count()
            splitIndex=data.columns.get_loc(self.split_column)
            classIndex=data.columns.get_loc(self.class_column)
            sortedData=data.sort_values(self.split_column).drop_duplicates(self.split_column)
            print("SMALLEST" + str(sortedData.iloc[0][self.split_column]-1)) 
            splits.append(sortedData.iloc[0][self.split_column]-1) #there should be nothing smaller than the first split point
            for i in range(sortedData[0].count()-1): #build in-betwe en splits
                if sortedData.iloc[i][self.class_column] != sortedData.iloc[i-1][self.class_column]:
                    splits.append((sortedData.iloc[i][self.split_column]+sortedData.iloc[i+1][self.split_column])/2) #todo: add example outcome checking

            splits.append(sortedData.iloc[-1][self.split_column]+1)#there should be nothing larger than the biggest split-add 1 to ensure
            # ltSplit = [] #at beginning there should be nothing in ltSplit, since its smaller than smallest split
            gtSplit=data #since the first split is smaller than all values everything is in gtSplit.
            for thisSplit in splits:
                ltSplit=data.loc[data[self.split_column]<thisSplit] #add values to ltSplit from gtSplit when they are less than the split
                gtSplit=data.loc[data[self.split_column]>=thisSplit]


                ltCount=ltSplit[0].count()
                gtCount=gtSplit[0].count()

                ltWeight=ltCount/numPoints
                gtWeight=gtCount/numPoints

                if ltCount==0:
                    ltEntropy=0
                else:
                    ltOutcomes=ltSplit[self.class_column].value_counts().apply(lambda x:x/ltCount)
                    ltEntropy=ltOutcomes.apply(lambda x:-x*math.log(x,2)).sum()

                if gtCount==0:
                    gtEntropy=0
                else:
                    gtOutcomes=gtSplit[self.class_column].value_counts().apply(lambda x:x/gtCount)
                    gtEntropy=gtOutcomes.apply(lambda x:-x*math.log(x,2)).sum()
                thisEntropy=ltWeight*ltEntropy+gtWeight*gtEntropy
                print("ENTROPY: "+ str(thisEntropy))
                entropy.append(thisEntropy)    
            cutVal = [splits[entropy.index(min(entropy))],min(entropy)]

        self.info_gain=parentEntropy-cutVal[1]

        self.point=cutVal[0]
    
    def partitions(self):
        '''Get the two partitions (child nodes) for this split.'''
        if self.partition_list:
            # This check ensures that the list is computed at most once.  Once computed 
            # it is stored
            return self.partition_list
        data = self.data
        split_column = self.split_column
        partition_list = []
        partition_list.append(data[data[split_column] <= self.point])
        partition_list.append(data[data[split_column] > self.point])
        self.partition_list = partition_list
        return partition_list

class DecisionTree(object):

    def __init__(self, max_depth=None):
        if (max_depth is not None and (max_depth != int(max_depth) or max_depth < 0)):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
        

    def fit(self,data, class_column):
        '''Fit a tree on data, in which class_column is the target.'''
        if (not isinstance(data, pd.DataFrame)):
            raise Exception("Invalid input")
        if (class_column not in data.columns):
            print(class_column)
            print(data.columns)
            raise Exception("Hello")
        self.data = data
        self.class_column = class_column
        self.non_class_columns = [c for c in data.columns if 
                                  c != class_column]
        self.root = self.recursive_build_tree(data, depth=0, name='0',parent_examples=None)
            
    def recursive_build_tree(self, data, depth, name,parent_examples):
        if depth<self.max_depth:
            if data[0].count()==0:
                tree=Node()
                tree.node_type="leaf"
                tree.data=data
                tree.label=parent_examples[self.class_column].mode().iloc[0]
                tree.name=name
                tree.children=[]
            elif len(data[self.class_column].unique())<=1: #if all data outcomes are identical then return the value
                tree=Node()
                tree.node_type="leaf"
                tree.data=data
                tree.label=data[self.class_column].iloc[0]
                tree.name=name
                tree.children=[]

            else: #since we want to be able to use attributes multiple times we do not track the attributes.
                A=self.importance(data)
                if set(A.data[A.split_column].unique())=={0,1}:
                    self.non_class_columns = [c for c in self.non_class_columns if c != A.split_column] 
                tree=Node()
                tree.split=A
                tree.data=data
                tree.name=name
                tree.children=[self.recursive_build_tree(thisPart, depth+1, name+"."+str(count),data) for count,thisPart in enumerate(A.partitions())]
            return tree
        else:
            if not data.empty:
                tree=Node()
                tree.node_type="leaf"
                tree.name=name
                tree.children=[]
                tree.data=data
                print("MODE: "+ str(data[self.class_column]))

                tree.label=data[self.class_column].mode().iloc[0]
                return tree

        
    
    def importance(self,data):
        attrImp=[]

        #iterate through all attributes and determine the one which returns smallest intropy gain
        for thisSplit in self.non_class_columns:    
            attrImp.append(Split(data,self.class_column,thisSplit))
        return max(attrImp,key=attrgetter('info_gain')) #otherwise return the attribute

    
    def predict(self, test):
        if self.root.node_type=='leaf':
            print("rows: "+str(list(test.index))+ " will will have value " + str(self.label))
            print('you need a better tree')
        rootTest=self.root.split
        splitCol=rootTest.split_column
        splitPoint=rootTest.point
        ltSplit=test[test[splitCol]<splitPoint]
        gtSplit=test[test[splitCol]>=splitPoint]
        if not ltSplit.empty:
            self.root.children[0].treeTraverse(ltSplit)
        if not gtSplit.empty:
            self.root.children[1].treeTraverse(gtSplit)

    
                
    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)
            
dsmall = df.iloc[0:10, list(range(3)) + [279]]
#dtest = df.iloc[110:120, list(range(20)) + [279]]
tree=DecisionTree(10)
tree.fit(dsmall,279)
tree.print()
#print(tree.predict(dtest))