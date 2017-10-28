import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("Data/Train_Dataset.csv")
target=train["new_label"]

relevant = train['mvar9'] != 0
train=train[relevant]
target=target[relevant] 
train["mvar41"] = train["mvar41"]+train["mvar40"]+train["mvar39"]+train["mvar38"]
train=train[['mvar41','mvar9']]

train = preprocessing.scale(train)

sm = SMOTE(random_state=12, ratio = 1.0)
train, target = sm.fit_sample(train, target)


from sklearn.svm import SVC
train_size = (int)(0.80 * len(train))
rf = SVC()
rf.fit(train[:train_size],target[:train_size])
print(rf.score(train[train_size:],target[train_size:]))
test_pred = rf.predict(train[train_size:])
