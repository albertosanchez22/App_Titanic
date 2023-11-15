import requests
url = 'http://localhost:5000/results'
r = requests.post(url,json={"PassengerId":20,"Pclass":1,"Age":93.0,"SibSp":1,"Parch":0,"Fare":12.0000,"C":1,"Q":0,"S":0,"male":1,"female":0})
#r = requests.post(url)
print("R= ",r)
print(r.json())