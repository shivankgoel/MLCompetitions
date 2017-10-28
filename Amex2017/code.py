import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("Data/Train_Dataset.csv")
target=train["new_label"]

train = train.drop("new_label",axis=1)
#train=train.drop("mvar3",axis=1)
#train=train.drop("mvar4",axis=1)
#train=train.drop("mvar10",axis=1)
#train=train.drop("mvar11",axis=1)
#train=train.drop("mvar14",axis=1)
#train=train.drop("mvar15",axis=1)
train=train.drop("label1",axis=1)
train=train.drop("label",axis=1)
#train["mvar41"] = train["mvar41"]+train["mvar40"]+train["mvar39"]+train["mvar38"]
#train=train.drop("mvar38",axis=1)
#train=train.drop("mvar39",axis=1)
#train=train.drop("mvar40",axis=1)

train = train.drop("mvar3",axis=1)
#relevant = [a and b for a, b in zip((train['mvar9'] != 0), (train["mvar3"] != 0))]
relevant = (train['mvar9'] != 0)
train=train[relevant]
target=target[relevant] 
#train = train.drop("mvar9",axis=1)

train = preprocessing.scale(train)

'''
Divide into test and train
'''
train_size = (int)(0.80 * len(train))
train1 = train
train = train1[:train_size]
train_labels = target[:train_size]
test = train1[train_size:]
test_labels = target[train_size:]


sm = SMOTETomek(smote = SMOTE(ratio = {'Yes':23891,'No':23891}))
train, train_labels = sm.fit_sample(train, train_labels)

#sm = SMOTE(random_state=None, ratio = 'auto')
#train, train_labels = sm.fit_sample(train, train_labels)


pca = PCA(n_components=28, whiten=True)
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)


from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=None, random_state=None)
#clf = tree.DecisionTreeClassifier()
rf = SVC()
clf.fit(train,train_labels)
print(clf.score(train,train_labels))
print(clf.score(test,test_labels))
test_pred = clf.predict(test)

