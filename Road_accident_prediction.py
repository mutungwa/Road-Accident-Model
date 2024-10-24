# Step 1: Install necessary libraries (if needed)
!pip install scikit-learn joblib

# Step 2: Upload the CSV dataset
from google.colab import files
import pandas as pd

# Upload the file from your local machine
uploaded = files.upload()

# Get the correct filename (use the actual filename of your uploaded file)
filename = list(uploaded.keys())[0]

# Step 3: Load the dataset (this time with the correct uploaded filename)
data = pd.read_csv(filename)

# Step 4: Preprocess the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
data['Weather'] = label_encoder.fit_transform(data['Weather'])
data['Road Condition'] = label_encoder.fit_transform(data['Road Condition'])
data['Time of Day'] = label_encoder.fit_transform(data['Time of Day'])

# Define independent and dependent variables
X = data[['Weather', 'Road Condition', 'Traffic Volume', 'Driver Age', 'Vehicle Age', 'Speed', 'Time of Day']]
y = data['Accident Severity']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a new Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Save the trained model
import joblib
joblib.dump(model, 'retrained_accident_severity_model.pkl')

# Step 8: Make predictions using a hypothetical scenario
hypothetical_data = pd.DataFrame({
    'Weather': [1],  # Encoded as 1 (Dry)
    'Road Condition': [1],  # Encoded as 1 (Dry)
    'Traffic Volume': [50],  # Example traffic volume
    'Driver Age': [35],  # Example driver age
    'Vehicle Age': [5],  # Example vehicle age
    'Speed': [80],  # Example speed
    'Time of Day': [0]  # Encoded as 0 (Day)
})

# Step 9: Predict accident severity using the model
predicted_severity = model.predict(hypothetical_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
