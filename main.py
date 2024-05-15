import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample test case data with features and labels
data = {
    'Number_of_Steps': [5, 10, 3, 8, 6, 12, 4],
    'Number_of_Inputs': [3, 5, 2, 4, 3, 6, 2],
    'Complexity_of_Logic': [2, 3, 1, 2, 2, 3, 1],
    'Complexity_Level': ['Simple', 'Complex', 'Simple', 'Medium', 'Simple', 'Complex', 'Simple']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df.drop('Complexity_Level', axis=1)
y = df['Complexity_Level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict complexity levels for test data
y_pred = clf.predict(X_test)

# Evaluate model performanc
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Now you can use this trained model to predict complexity levels for new test cases
# For example:
new_test_case = [[5, 2, 12]]  # Features: [Number_of_Steps, Number_of_Inputs, Complexity_of_Logic]
predicted_complexity = clf.predict(new_test_case)
print("Predicted Complexity Level for new test case:", predicted_complexity)
