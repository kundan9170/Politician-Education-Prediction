import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
from sklearn.preprocessing import LabelEncoder

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

# Function to preprocess 'Constituency ∇' field
def preprocess_constituency(value):
    if value.endswith('(SC)'):
        return 'SC'
    elif value.endswith('(ST)'):
        return 'ST'
    else:
        return 'GEN'

# Function to group states into East, West, North, and South
def group_states_by_region(state):
    state = state.upper()  # Convert state name to uppercase
    east_states = ['BIHAR', 'JHARKHAND', 'ODISHA', 'WEST BENGAL', 'SIKKIM', 'ASSAM', 'ARUNACHAL PRADESH', 'MANIPUR', 'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'TRIPURA']
    west_states = ['RAJASTHAN', 'GUJARAT', 'GOA', 'MAHARASHTRA']
    north_states = ['JAMMU AND KASHMIR', 'HIMACHAL PRADESH', 'PUNJAB', 'HARYANA', 'UTTAR PRADESH', 'UTTARAKHAND', 'DELHI', 'CHANDIGARH']
    south_states = ['KERALA', 'KARNATAKA', 'TAMIL NADU', 'ANDHRA PRADESH', 'TELANGANA', 'PUDUCHERRY', 'ANDAMAN AND NICOBAR ISLANDS', 'LAKSHADWEEP']
    
    if state in east_states:
        return 1
    elif state in west_states:
        return 2
    elif state in north_states:
        return 3
    elif state in south_states:
        return 6
    else:
        return 1

 # Function to preprocess Candidate Name field
def preprocess_candidate(value):
    if value.startswith('Dr.'):
        return 2
    elif value.startswith('Adv.'):
        return 1
    else:
        return 0

# Load the dataset
data = pd.read_csv('train.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'Education' field using label encoding
data['Education'] = data['Education'].apply(map_education_to_numeric)
data['Candidate'] = data['Candidate'].apply(preprocess_candidate)

# Group states into regions
data = pd.get_dummies(data, columns=['state'])

# One-hot encode the 'Party' field in test data
data = pd.get_dummies(data, columns=['Party'])

# Preprocess 'Constituency ∇' field
data['Constituency ∇'] = data['Constituency ∇'].apply(preprocess_constituency)
data = pd.get_dummies(data, columns=['Constituency ∇'])

# Remove rows with any missing entries (NaN or blank)
data.dropna(inplace=True)

# Remove the 'ID', 'Candidate', 'state', and 'Education' columns
data.drop(columns=['ID'], inplace=True)

# Convert 'Total Assets' and 'Liabilities' columns to numerical values
data['Total Assets'] = data['Total Assets'].apply(convert_to_numeric)
data['Liabilities'] = data['Liabilities'].apply(convert_to_numeric)

# correlation_matrix = data.corrwith(data['Education'])
# # Print the correlation matrix

# # Save the correlation matrix as a text file
# correlation_matrix.to_csv('correlation_matrix.txt', sep='\t')
# print("Correlation Matrix:")
# print(correlation_matrix)

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=['Education'])
y = data['Education']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest Classifier with custom parameters
n_estimators_value = 100  # Adjust the number of estimators as needed
random_forest_model = RandomForestClassifier(n_estimators=n_estimators_value)

# Train the model using the training dataset
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
test_predictions = random_forest_model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:", accuracy)


# Calculate F1 score
f1 = f1_score(y_test, test_predictions, average='weighted')
print("F1 Score:", f1)


# Now, let's use the entire dataset for training
# Initialize the RandomForest Classifier with custom parameters
random_forest_model_full = RandomForestClassifier(n_estimators=n_estimators_value)

# Train the model using the entire dataset
random_forest_model_full.fit(X, y)

# Load the test dataset
test_data = pd.read_csv('test.csv')

# One-hot encode the 'state' field in test data
test_data = pd.get_dummies(test_data, columns=['state'])

# One-hot encode the 'Party' field in test data
test_data = pd.get_dummies(test_data, columns=['Party'])

# Convert 'Total Assets' and 'Liabilities' columns to numerical values
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_to_numeric)
test_data['Liabilities'] = test_data['Liabilities'].apply(convert_to_numeric)

# Taking into account Suffix of the names.
test_data['Candidate'] = test_data['Candidate'].apply(preprocess_candidate)

# Remove the 'ID', 'Candidate', and 'Constituency ∇' columns
test_data.drop(columns=['ID'], inplace=True)

# Preprocess 'Constituency ∇' field
test_data['Constituency ∇'] = test_data['Constituency ∇'].apply(preprocess_constituency)
test_data = pd.get_dummies(test_data, columns=['Constituency ∇'])

# Make predictions for test data
test_predictions = random_forest_model_full.predict(test_data)

#Convert predictions to education categories
test_predictions_mapped = [map_numeric_to_education(numeric) for numeric in test_predictions]

# #Create a DataFrame for test predictions
predictions_df = pd.DataFrame({'ID': range(len(test_predictions_mapped)), 'Education': test_predictions_mapped})

# Save predictions to a CSV file
predictions_df.to_csv('submission_final_final.csv', index=False)

print("Predictions saved to final_submission.csv")
