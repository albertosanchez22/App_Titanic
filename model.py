import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle


titanic_training = pd.read_csv("./train.csv")
titanic_test = pd.read_csv("./test.csv")
titanic_sinNan = titanic_training.dropna()

SEED = 42
X_transformada_Sex = pd.get_dummies(titanic_sinNan.Sex)
X_transformada_Embark = pd.get_dummies(titanic_sinNan.Embarked)
X = titanic_sinNan.drop(columns=["Name","Sex","Ticket", "Cabin", "Embarked", "Survived"])
X["C"] = X_transformada_Embark["C"]
X["Q"] = X_transformada_Embark["Q"]
X["S"] = X_transformada_Embark["S"]
X["male"] = X_transformada_Sex["male"]
X["female"] = X_transformada_Sex["female"]
y = titanic_sinNan["Survived"].to_numpy()

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)


lir = LogisticRegression(max_iter=500)
ada = AdaBoostClassifier(n_estimators=150, random_state = SEED)
bagg = BaggingClassifier(estimator=SVC(), n_estimators=25, random_state=0).fit(X, y)
rf = RandomForestClassifier(max_depth=3, random_state=0)

rf.fit(X_train, y_train)
print("Random Forest Classifier: {:.3f}" .format(rf.score(X_test, y_test)))
lir.fit(X_train, y_train)
print("Logistic Classifier: {:.3f}" .format(lir.score(X_test, y_test)))
ada.fit(X_train, y_train)
print("Ada Boost Regressor: {:.3f}" .format(ada.score(X_test, y_test)))
bagg.fit(X_train, y_train)
print("Bagging Regressor: {:.3f}" .format(bagg.score(X_test, y_test)))

classifiers = [('Logistic Classifier', lir),
               ('Random Forest Regressor', rf),
               ('Ada Boost Classifier', ada)]

for clf_name, clf in classifiers:
  clf.fit(X_train, y_train)
  #print("{:s}: {:.3f}" .format(clf_name, clf.score(X_test, y_test)))

vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
print("Voting Classifier: {:.3f}" .format(vc.score(X_test, y_test)))

pickle.dump(vc, open('model.pkl','wb'))


