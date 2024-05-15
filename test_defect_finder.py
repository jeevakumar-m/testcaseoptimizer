# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Simulate a dataset resembling SAP / e-commerce applications
data = {
    'User_ID': ['user1', 'user2', 'user3', 'user4', 'user5'] * 200,
    'Product_ID': ['prod1', 'prod2', 'prod1', 'prod3', 'prod2'] * 200,
    'Page_Viewed': ['Home', 'Product', 'Cart', 'Checkout', 'Confirmation'] * 200,
    'Action': ['Click', 'Add_to_Cart', 'Remove_from_Cart', 'Purchase', 'Logout'] * 200,
    'Time_Spent': [10, 20, 5, 15, 2] * 200,
    'Date': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
    'Hour_of_Day': [d.hour for d in pd.date_range(start='2023-01-01', periods=1000, freq='D')],
    'Day_of_Week': [d.dayofweek for d in pd.date_range(start='2023-01-01', periods=1000, freq='D')],
    'Month': [d.month for d in pd.date_range(start='2023-01-01', periods=1000, freq='D')],
    'Defect_Prone_Zone': [0, 1, 0, 1, 0] * 200  # 0 indicates not defect-prone, 1 indicates defect-prone
}

# Convert data to DataFrame
defect_data = pd.DataFrame(data)

# Define features and target variable
X = defect_data.drop(columns=["Defect_Prone_Zone"])
y = defect_data["Defect_Prone_Zone"]

# Convert categorical variables to numerical representations using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example of using the trained model to predict defect-prone zones in new data
# Simulate new data for prediction
new_data = pd.DataFrame({
    'User_ID': ['user1'],
    'Product_ID': ['prod1'],
    'Page_Viewed': ['Home'],
    'Action': ['Click'],
    'Time_Spent': [10],
    'Date': ['2023-06-10'],
    'Hour_of_Day': [15],
    'Day_of_Week': [3],
    'Month': [6]
})

# Convert categorical variables to numerical representations using one-hot encoding
new_data = pd.get_dummies(new_data)

# Ensure the new data has the same columns as the training data
missing_cols = set(X_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0

# Reorder columns to match the order of training data
new_data = new_data[X_train.columns]

# Predict defect-prone zones in new data
predicted_zones = rf_classifier.predict(new_data)
print("Predicted Defect-Prone Zone:", predicted_zones[0])
