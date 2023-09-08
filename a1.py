import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_excel(r"C:\z_space\5th sem\ML\ML-lab2-main\19CSE305_LabData_Set3.1.xlsx", sheet_name='thyroid0387_UCI')
data_types = data.dtypes

categorical_cols = data.select_dtypes(include=['object']).columns
nominal_cols = ['referral source'] + [col for col in data.columns if data[col].dtype == 'O' and data[col].str.contains('\?').any()]
ordinal_cols = list(set(categorical_cols) - set(nominal_cols))

numeric_cols = data.select_dtypes(include=['number'])
data_range = numeric_cols.describe().loc[['min', 'max']]

missing_values = data.isna().sum()

outliers = {}
for col in numeric_cols.columns:
    mean = numeric_cols[col].mean()
    std = numeric_cols[col].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers[col] = len(numeric_cols[(numeric_cols[col] < lower_bound) | (numeric_cols[col] > upper_bound)])

numeric_mean = numeric_cols.mean()
numeric_variance = numeric_cols.var()

print("Task 1: Data Types")
print(data_types)

print("\nTask 2: Encoding Schemes")
print("Nominal Columns:", nominal_cols)
print("Ordinal Columns:", ordinal_cols)

print("\nTask 3: Data Range")
print(data_range)

print("\nTask 4: Missing Values")
print(missing_values)

print("\nTask 5: Outliers")
print(outliers)

print("\nTask 6: Mean and Variance for Numeric Variables")
print("Mean:")
print(numeric_mean)
print("\nVariance:")
print(numeric_variance)

# Fill missing values based on data type and presence of outliers
for col in data.columns:
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        # Numeric attribute
        if col in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
            # Use median for attributes with outliers
            data=data[col].fillna(data[col].median(), inplace=True)
        else:
            # Use mean for attributes without outliers
            data=data[col].fillna(data[col].mean(), inplace=True)
    elif data[col].dtype == 'object':
        data=data[col].fillna(data[col].mode()[0], inplace=True)

missing_values_after_imputation = data.isnull().sum()

print("Missing Values After Imputation:")
print(missing_values_after_imputation)


numeric_attributes = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

minmax_scaler = MinMaxScaler()
data[numeric_attributes] = minmax_scaler.fit_transform(data[numeric_attributes])

print("Normalized Data:")
print(data.head())

# Extract the binary attributes (assuming binary attributes are 'on thyroxine' to 'hypopituitary')
binary_attributes = ['onthyroxine', 'queryonthyroxine', 'onantithyroidmedication',
                     'sick', 'pregnant', 'thyroidsurgery', 'I131treatment', 'queryhypothyroid',
                     'queryhyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary']

# Extract the first two observation vectors
vector1 = data.loc[0, binary_attributes].astype(str)
vector2 = data.loc[1, binary_attributes].astype(str)

# Calculate f11, f01, f10, f00
f11 = sum((vector1 == '1') & (vector2 == '1'))
f01 = sum((vector1 == '0') & (vector2 == '1'))
f10 = sum((vector1 == '1') & (vector2 == '0'))
f00 = sum((vector1 == '0') & (vector2 == '0'))

# Calculate Jaccard Coefficient (JC) if denominator is not zero
if f01 + f10 + f11 != 0:
    jc = f11 / (f01 + f10 + f11)
else:
    jc = 0.0  # Set JC to 0 if denominator is zero

# Calculate Simple Matching Coefficient (SMC) if denominator is not zero
if f00 + f01 + f10 + f11 != 0:
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
else:
    smc = 0.0  # Set SMC to 0 if denominator is zero

# Print the results
print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)

# Extract the feature vectors for the first two observations
vector_1 = data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']].astype(float)  # Using all attributes except the first column (Record ID)
vector_2 = data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']].astype(float)  # Using all attributes except the first column (Record ID)

# Calculate the dot product of the two vectors
dot_product = np.dot(vector_1, vector_2)

# Calculate the magnitude (length) of each vector
magnitude_vector1 = np.linalg.norm(vector_1)
magnitude_vector2 = np.linalg.norm(vector_2)

# Calculate the cosine similarity
cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)

# Print the cosine similarity
print("Cosine Similarity:", cosine_similarity)

# Extract the first 20 observation vectors
vectors = data.iloc[:20, 1:-1]  # Exclude the first column (Record ID) and the last column (Condition)

# Define a function to calculate Jaccard Coefficient
def jaccard_coefficient(vector_1, vector_2):
    intersection = np.logical_and(vector_1, vector_2)
    union = np.logical_or(vector_1, vector_2)
    return np.sum(intersection) / np.sum(union)

# Initialize matrices to store JC, SMC, and Cosine Similarity values
jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cosine_matrix = np.zeros((20, 20))

# Calculate JC, SMC, and Cosine Similarity between vectors
for i in range(20):
    for j in range(20):
        vector_1 = vectors.iloc[i].astype(bool)
        vector_2 = vectors.iloc[j].astype(bool)
        jc_matrix[i, j] = jaccard_coefficient(vector_1, vector_2)
        smc_matrix[i, j] = jaccard_score(vector_1, vector_2, average='binary')
        cosine_matrix[i, j] = cosine_similarity([vector_1], [vector_2])[0, 0]

# Create a heatmap for JC
plt.figure(figsize=(10, 8))
sns.heatmap(jc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Jaccard Coefficient Heatmap")
plt.show()

# Create a heatmap for SMC
plt.figure(figsize=(10, 8))
sns.heatmap(smc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Simple Matching Coefficient Heatmap")
plt.show()

# Create a heatmap for Cosine Similarity
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity Heatmap")
plt.show()


# Select the first 2 observation vectors
selected_data = data.iloc[:2, 1:-1]  # Exclude the first column (ID) and the last column (Response)

# Fill missing values '?' with 0 (assuming '?' means 'No' or 'False')
selected_data = selected_data.replace('?', 0)

# Identify columns with non-numeric values
non_numeric_columns = selected_data.select_dtypes(exclude=[np.number]).columns

# Convert selected_data to numeric, handling non-numeric columns separately
for column in non_numeric_columns:
    selected_data[column] = pd.to_numeric(selected_data[column], errors='coerce')

# Fill NaN values with 0
selected_data = selected_data.fillna(0)

# Calculate the Cosine similarity between the two vectors
cosine_sim = cosine_similarity(selected_data)

# Print the Cosine similarity matrix
print("Cosine Similarity Matrix:")
print(cosine_sim)
