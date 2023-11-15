import requests
url = 'http://localhost:5000/results'
r = requests.post(url,json={"PassengerId":20,"Pclass":1,"Age":25,"SibSp":1,"Parch":0,"Fare":12,"C":1,"Q":0,"S":0,"male":0,"female":1})
print("R= ",r)
print(r.json())