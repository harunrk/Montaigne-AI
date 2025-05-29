import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("Train_düzenlenmiş_veri.csv")
test = pd.read_csv("Test_düzenlenmiş_veri.csv")

train = train.drop(["PassengerId","Cabin",'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',"Surname"], axis=1)
passengerIds = test["PassengerId"]
test = test[train.drop("Transported",axis=1).columns]

x = train.drop("Transported", axis=1)
y = train["Transported"].values

ohe = OneHotEncoder()

homeplanet = x[["HomePlanet"]]
homeplanet = ohe.fit_transform(homeplanet).toarray()
homeplanet = np.delete(homeplanet, 0, axis=1)
x = x.drop(["HomePlanet"], axis=1)
x = pd.concat([x, pd.DataFrame(homeplanet)], axis=1)

homeplanet = test[["HomePlanet"]]
homeplanet = ohe.transform(homeplanet).toarray()
homeplanet = np.delete(homeplanet, 0, axis=1)
test = test.drop(["HomePlanet"], axis=1)
test = pd.concat([test, pd.DataFrame(homeplanet)], axis=1)

destination = x[["Destination"]]
destination = ohe.fit_transform(destination).toarray()
destination = np.delete(destination, 0, axis=1)
x = x.drop(["Destination"], axis=1)
x = pd.concat([x, pd.DataFrame(destination)], axis=1)

destination = test[["Destination"]]
destination = ohe.transform(destination).toarray()
destination = np.delete(destination, 0, axis=1)
test = test.drop(["Destination"], axis=1)
test = pd.concat([test, pd.DataFrame(destination)], axis=1)

CabinDeck = x[["Cabin_deck"]]
CabinDeck = ohe.fit_transform(CabinDeck).toarray()
CabinDeck = np.delete(CabinDeck, 0, axis=1)
x = x.drop(["Cabin_deck"], axis=1)
x = pd.concat([x, pd.DataFrame(CabinDeck)], axis=1)

CabinDeck = test[["Cabin_deck"]]
CabinDeck = ohe.transform(CabinDeck).toarray()
CabinDeck = np.delete(CabinDeck, 0, axis=1)
test = test.drop(["Cabin_deck"], axis=1)
test = pd.concat([test, pd.DataFrame(CabinDeck)], axis=1)

cabinSide = x[["Cabin_side"]]
cabinSide = ohe.fit_transform(cabinSide).toarray()
cabinSide = np.delete(cabinSide, 0, axis=1)
x = x.drop(["Cabin_side"], axis=1)
x = pd.concat([x, pd.DataFrame(cabinSide)], axis=1)

cabinSide = test[["Cabin_side"]]
cabinSide = ohe.transform(cabinSide).toarray()
cabinSide = np.delete(cabinSide, 0, axis=1)
test = test.drop(["Cabin_side"], axis=1)
test = pd.concat([test, pd.DataFrame(cabinSide)], axis=1)

le = LabelEncoder()
x["CryoSleep"] = le.fit_transform(x["CryoSleep"])
test["CryoSleep"] = le.transform(test["CryoSleep"])

x["VIP"] = le.fit_transform(x["VIP"])
test["VIP"] = le.transform(test["VIP"])

del cabinSide, CabinDeck, homeplanet, destination, ohe, le

# xtrain, xtest, ytrain, ytest = train_test_split(x.values, y.ravel(), test_size=0.33, random_state=0)

# DTC -------------------------------------------------------------------
# dtc = DecisionTreeClassifier(criterion="log_loss") %69
# dtc.fit(xtrain, ytrain)
# ypred = dtc.predict(xtest)

# KNN -------------------------------------------------------------------
# knn = KNeighborsClassifier(n_neighbors=23, metric="minkowski") # %73.3008
# knn.fit(xtrain, ytrain)
# ypred = knn.predict(xtest)

# results = []
# for i in range(1, 30):
#     knn = KNeighborsClassifier(n_neighbors=i, metric="minkowski")
#     knn.fit(xtrain, ytrain)
#     ypred = knn.predict(xtest)
#     cm = confusion_matrix(ytest, ypred)
#     percentOf = round(((cm[0,0] + cm[1,1])/ (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]) * 100), 4)

#     results.append(percentOf)

# plt.plot(range(len(results)), results)
# plt.show()

# LRG -------------------------------------------------------------------
# logR = LogisticRegression(random_state=0) # %70.8958
# logR.fit(xtrain, ytrain)
# ypred = logR.predict(xtest)

# RFC -------------------------------------------------------------------

# rfc = RandomForestClassifier(n_estimators=14, criterion="gini") # max %73.8236 
# rfc.fit(xtrain, ytrain)
# ypred = rfc.predict(xtest)

# results = []
# for i in range(1, 30):
#     rfc = RandomForestClassifier(n_estimators=i, criterion="gini")
#     rfc.fit(xtrain, ytrain)
#     ypred = rfc.predict(xtest)
#     cm = confusion_matrix(ytest, ypred)
#     percentOf = round(((cm[0,0] + cm[1,1])/ (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]) * 100), 4)
#     results.append(percentOf)

# plt.plot(range(len(results)), results)
# plt.show()

# print("Accuracy: ", percentOf)

# cm = confusion_matrix(ytest, ypred)
# percentOf = round(((cm[0,0] + cm[1,1])/ (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]) * 100), 4)
# print("Accuracy: ", percentOf)

rfc = RandomForestClassifier(n_estimators=14, criterion="gini") # max %73.8236 
rfc.fit(x.values, y)
ypred = rfc.predict(test.values)
ypred = pd.DataFrame(data=ypred, columns=["Transported"])
ypred.replace(0, False, inplace=True)
ypred.replace(1, True, inplace=True)

ypred = pd.concat([passengerIds, ypred], axis=1)

ypred.to_csv("submission.csv", index=False)







