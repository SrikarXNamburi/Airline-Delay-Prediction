import pandas as pd
penguins = pd.read_csv('Airlines.csv')

df = penguins.copy()
df = df.drop('id',axis=1)
df = df.drop('Flight',axis=1)
target = 'Delay'
encode = ['Airline','AirportFrom','AirportTo']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Separating X and y
X = df.drop('Delay', axis=1)
Y = df['Delay']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('airlines_model.pkl', 'wb'))