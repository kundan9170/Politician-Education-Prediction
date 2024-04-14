from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Function to map education categories to numerical values
def map_education_to_numeric(education):
    education_mapping = {
        '8th Pass': 6,
        '10th Pass': 8,
        '12th Pass': 10,
        'Graduate': 14,
        'Graduate Professional': 15,
        'Post Graduate': 17,
        'Doctorate': 20,
        'Literate': 4,
        '5th Pass': 5,
        'Others': 2
    }
    return education_mapping.get(education, 0)  # Return 0 for unknown categories

# Function to map numerical values back to education categories
def map_numeric_to_education(numeric):
    numeric_mapping = {
        6: '8th Pass',
        8: '10th Pass',
        10: '12th Pass',
        14: 'Graduate',
        15: 'Graduate Professional',
        17: 'Post Graduate',
        20: 'Doctorate',
        4: 'Literate',
        5: '5th Pass',
        2: 'Others'
    }
    return numeric_mapping.get(numeric, 'Unknown')  # Return 'Unknown' for unknown values

# Function to convert 'Total Assets' and 'Liabilities' columns to numerical values
def convert_to_numeric(value):
    if 'Crore+' in value:
        return float(value.split()[0]) * 10000000
    elif 'Lac+' in value:
        return float(value.split()[0]) * 100000
    elif 'Thou+' in value:
        return float(value.split()[0]) * 1000
    elif 'Hund+' in value:
        return float(value.split()[0]) * 100
    else:
        return float(value)

# Function to preprocess 'Constituency' field
def preprocess_constituency(value):
    if value.endswith('(SC)'):
        return 'SC'
    elif value.endswith('(ST)'):
        return 'ST'
    else:
        return 'GEN'

# Function to preprocess 'Candidate' field
def preprocess_candidate(value):
    if value.startswith('Dr.'):
        return 2
    elif value.startswith('Adv.'):
        return 1
    else:
        return 0

# Load the dataset
data = pd.read_csv('train.csv')

# Encode the 'Education' field using numerical values
data['Education'] = data['Education'].apply(map_education_to_numeric)

# Preprocess 'Candidate' field
data['Candidate'] = data['Candidate'].apply(preprocess_candidate)

# One-hot encode the 'state' and 'Party' fields
data = pd.get_dummies(data, columns=['state', 'Party'])

# Preprocess 'Constituency' field
data['Constituency ∇'] = data['Constituency ∇'].apply(preprocess_constituency)
data = pd.get_dummies(data, columns=['Constituency ∇'])

# Remove rows with any missing entries
data.dropna(inplace=True)

# Remove the 'ID' column
data.drop(columns=['ID'], inplace=True)

# Convert 'Total Assets' and 'Liabilities' columns to numerical values
data['Total Assets'] = data['Total Assets'].apply(convert_to_numeric)
data['Liabilities'] = data['Liabilities'].apply(convert_to_numeric)

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=['Education'])
y = data['Education']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN Classifier with custom parameters
knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Train the model using the training dataset
knn_model.fit(X_train, y_train)

# Make predictions on the test data
test_predictions = knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(y_test, test_predictions, average='weighted')
print("F1 Score:", f1)
