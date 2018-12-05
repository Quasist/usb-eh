import pandas as pd
import numpy as np
import csv
from sklearn import cross_validation
from sklearn.svm import LinearSVC

#load data
data = data.csv
names = ['a', 'b', 'c', 'b 3', 'b micro 3', 'b mini', 'b micro']
dataset = pd.read_csv(data, names=names)

# head
print(dataset.head(20))

#classifier
clf = LinearSVC(gamma=0.001, C=100.)
