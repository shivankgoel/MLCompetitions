import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("Data/Train_Dataset.csv")
test  = pd.read_csv("Data/Leaderboard_Dataset.csv")

target=train["new_label"]
train = train.drop("new_label",axis=1)
train=train.drop("mvar3",axis=1)
train=train.drop("mvar4",axis=1)
train=train.drop("mvar10",axis=1)
train=train.drop("mvar11",axis=1)
train=train.drop("mvar14",axis=1)
train=train.drop("mvar15",axis=1)
train=train.drop("label1",axis=1)
train=train.drop("label",axis=1)

test=test.drop("mvar3",axis=1)
test=test.drop("mvar4",axis=1)
test=test.drop("mvar10",axis=1)
test=test.drop("mvar11",axis=1)
test=test.drop("mvar14",axis=1)
test=test.drop("mvar15",axis=1)
train["mvar41"] = train["mvar41"]+train["mvar40"]+train["mvar39"]+train["mvar38"]
test["mvar41"] = test["mvar41"]+test["mvar40"]+test["mvar39"]+test["mvar38"]

train=train.drop("mvar38",axis=1)
train=train.drop("mvar39",axis=1)
train=train.drop("mvar40",axis=1)

test=test.drop("mvar38",axis=1)
test=test.drop("mvar39",axis=1)
test=test.drop("mvar40",axis=1)
test=test.drop("mvar1",axis=1)
test=test.drop("mvar12",axis=1)


train = preprocessing.scale(train)
test = preprocessing.scale(test)


pca = PCA(n_components=10, whiten=True)
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)
from sklearn.svm import SVC
train_size = (int)(0.80 * len(train))
rf = SVC()
rf.fit(train[:train_size],target[:train_size])
print(rf.score(train[train_size:],target[train_size:]))
Y_Pred = rf.predict(test)

submission = pd.DataFrame(Y_Pred, columns=['Label'], index=np.arange(1, 10001))
submission.to_csv("result.csv")