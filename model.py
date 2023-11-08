import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression



titanic_training = pd.read_csv(r"C:\Users\BigData\Desktop\CursoBigData\csvTitanic\train.csv")
titanic_test = pd.read_csv(r"C:\Users\BigData\Desktop\CursoBigData\csvTitanic\test.csv")
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
survived = titanic_sinNan.drop(columns=["Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Pclass", "PassengerId"])
y = survived

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)


lir = LogisticRegression()

lir.fit(X_train, y_train)
print("Linear Regression: {:.3f}" .format(lir.score(X_test, y_test)))

