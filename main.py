from PIL import Image
import numpy, os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
path="frames/"
Xlist=[]
Ylist=[]
for directory in os.listdir(path):
        for directory2 in os.listdir(path+directory):
                for file in os.listdir(path+directory+"/"+directory2):
                        print(path+directory+"/"+directory2+"/"+file)
                        img=Image.open(path+directory+"/"+directory2+"/"+file)
                        featurevector=numpy.array(img).flatten()[:50] #in my case the images dont have the same dimensions, so [:50] only takes the first 50 values
                        Xlist.append(featurevector)
                        Ylist.append(directory)
clf=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, Xlist, Ylist)
print(scores.mean())
