## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```

# EXPERIMENT 3: FEATURE ENCODING AND FEATURE TRANSFORMATION

# Step 1: Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy import stats

# =====================================================
# PART A: FEATURE ENCODING
# =====================================================

# Step 2: Read the encoding datasets

encoding_data = pd.read_csv("Encoding Data.csv")
data = pd.read_csv("data.csv")

print("Encoding Dataset:")
print(encoding_data.head())

print("\nData Dataset:")
print(data.head())


# Step 3: Data Cleaning

# Fill missing numerical values with mean
for df in [encoding_data, data]:
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for df in [encoding_data, data]:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])


# Step 4: Feature Encoding for Encoding Data.csv

# Binary Encoding using Label Encoder
le = LabelEncoder()

encoding_data['bin_1'] = le.fit_transform(encoding_data['bin_1'])
encoding_data['bin_2'] = le.fit_transform(encoding_data['bin_2'])

# Ordinal Encoding
ord_2_mapping = {
    'Cold': 0,
    'Warm': 1,
    'Hot': 2
}

encoding_data['ord_2'] = encoding_data['ord_2'].map(ord_2_mapping)

# Nominal Encoding using One-Hot Encoding
encoding_data = pd.get_dummies(
    encoding_data,
    columns=['nom_0'],
    dtype=int
)

print("\nEncoded Encoding Data:")
print(encoding_data.head())


# Step 5: Feature Encoding for data.csv

# Binary Encoding
data['bin_1'] = le.fit_transform(data['bin_1'])
data['bin_2'] = le.fit_transform(data['bin_2'])

# Ordinal Encoding
ord_1_mapping = {
    'Cold': 0,
    'Warm': 1,
    'Hot': 2,
    'Very Hot': 3
}

ord_2_mapping = {
    'High School': 0,
    'Diploma': 1,
    'Bachelors': 2,
    'Masters': 3,
    'PhD': 4
}

data['Ord_1'] = data['Ord_1'].map(ord_1_mapping)
data['Ord_2'] = data['Ord_2'].map(ord_2_mapping)

# Nominal Encoding using One-Hot Encoding
data = pd.get_dummies(
    data,
    columns=['City'],
    dtype=int
)

print("\nEncoded Data:")
print(data.head())


# Step 6: Save Encoded Data

encoding_data.to_csv("Encoded_Data.csv", index=False)
data.to_csv("Processed_Data.csv", index=False)

print("\nFeature Encoding Completed Successfully!")


# =====================================================
# PART B: FEATURE TRANSFORMATION
# =====================================================

# Step 7: Read the transformation dataset

fare_data = pd.read_csv("Data_to_Transform.csv")

print("\nOriginal Transformation Dataset:")
print(fare_data.head())

# Step 8: Data Cleaning

fare_data = fare_data.fillna(fare_data.mean())

# Create a separate dataframe for transformed features
transformed_data = fare_data.copy()

# Step 9: Apply Feature Transformations

for col in fare_data.columns:

    # Shift negative values to make all values positive
    shift = max(0, 1 - fare_data[col].min())

    x = fare_data[col] + shift

    # Log Transformation
    transformed_data[col + '_log'] = np.log(x)

    # Reciprocal Transformation
    transformed_data[col + '_reciprocal'] = 1 / x

    # Square Root Transformation
    transformed_data[col + '_sqrt'] = np.sqrt(x)

    # Square Transformation
    transformed_data[col + '_square'] = np.square(x)

    # Box-Cox Transformation
    transformed_data[col + '_boxcox'], _ = stats.boxcox(x)

# Step 10: Yeo-Johnson Transformation

pt = PowerTransformer(method='yeo-johnson')

yeo_data = pt.fit_transform(fare_data)

yeo_df = pd.DataFrame(
    yeo_data,
    columns=[col + '_yeojohnson' for col in fare_data.columns]
)

transformed_data = pd.concat(
    [transformed_data, yeo_df],
    axis=1
)

# Step 11: Display transformed data

print("\nTransformed Dataset:")
print(transformed_data.head())

# Step 12: Save transformed data

transformed_data.to_csv(
    "Transformed_Data.csv",
    index=False
)

print("\nFeature Transformation Completed Successfully!")

print("\nAll processed files saved successfully!")

```

<img width="597" height="796" alt="image" src="https://github.com/user-attachments/assets/68999c93-e563-4bca-88b0-7c551bd086c5" />
<img width="600" height="798" alt="image" src="https://github.com/user-attachments/assets/4166a1a0-8a52-4855-bb4d-193d2af94467" />
<img width="600" height="132" alt="image" src="https://github.com/user-attachments/assets/e125e54c-fe54-4b98-818d-b8c06dca22f3" />



# RESULT:
      Thus, the program to implement Feature Encoding and Feature Transformation was successfully written and verified using Python programming.

       
