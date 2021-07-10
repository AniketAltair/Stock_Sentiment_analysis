import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle

df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
#print(df.head())

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# remove unwanted characters like :,.? and replace them by " "
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
#print(data)

# Renaming col name from Top1,.. to 1,2...
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
#print(data)

# Convert to lower case all data, as we will be using bag of words, so Good and good will be treated seperately if not done, which we don't want
for index in new_Index:
    data[index]=data[index].str.lower()

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#print(headlines[1])

countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
#randomclassifier.fit(traindataset,train['Label'])

"""filename = 'trained_model.sav'
pickle.dump(randomclassifier, open(filename, 'wb'))"""

loaded_model = pickle.load(open("trained_model.sav", 'rb'))
print("################")
print(loaded_model)
print("################")

test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = loaded_model.predict(test_dataset)

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)