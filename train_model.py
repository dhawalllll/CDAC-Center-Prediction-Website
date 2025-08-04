import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Dummy data: ranks vs center type
data = {
    'rank': [100, 250, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000],
    'center_type': ['CDAC ACTS HQ and Sunbeam Pune', 'Sunbeam Pune', 'CDAC ACTS HQ and Sunbeam Pune', 'CDAC ACTS HQ, Sunbeam Pune ,CDAC Mumbai or CDAC Banglore(IACSD)' ,'CDAC ACTS HQ, Sunbeam pune,CDAC Mumbai or CDAC Banglore', 'CDAC Pune(ACTS), CDAC Banglore, and  Sunbeam Pune', 'IACSD(Akurdi) and Sunbeam Karad', 'CDAC Pune(ACTS),CDAC Mumbai and Sunbeam Pune', 'CDAC Hyderabad,CDAC Noida/Chennai/Kochi/Kolkata', 'CDAC Mumbai,CDAC Banglore,IACSD Pune or VITA Mumbai']
    
    
}

df = pd.DataFrame(data)
X = df[['rank']]
y = df['center_type']

# Train a simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
