import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

with open('ML_Part2.ml', 'rb') as f:
    classifier = pickle.load(f)

df = pd.read_csv("churning.csv")
# we use only the first 2 variables as features
# account length,number vmail messages
x = df.iloc[:, [0, 1]]
y = df.iloc[:,-1] # classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))