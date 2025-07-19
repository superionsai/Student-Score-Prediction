import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
df = pd.read_csv('StudentsPerformance.csv')
# Importing the dataset
df
#Analysing the columns in the dataset
for cols in df.columns:
    if df[cols].dtype == 'object':
        print(f"{cols} is a categorical variable")
        print(df[cols].unique())
    else:
        print(f"{cols} is a numerical variable")
        print(df[cols].unique())
"""
We have checked the columns and identified the type and flaws.
What we can see in parental level of education is that there are so many random values like 'some high school', 'some college'.
We have to convert some high school to high school and preferably some college to bachelor's degree.
"""
#Now we shall check for null values in the dataset
df.isnull().sum()
""" There are no null values in the dataset.
Now we shall check for duplicate values in the dataset."""
df.duplicated().sum()
#No duplicate values in the dataset.
""" We have checked the columns and identified the data types, since there is no necessity for cleaning the data due to no presence of null values and duplicates"""
#We can check for outliers in the dataset and remove them, according to the scores. This is because the scores like 0 or 8 are too low and rather exceptional.
#There could be cases like absence or some other reason that could alter our predictions.
#DATA ANALYSIS
fig, axes = plt.subplots(1, 3, figsize=(8, 4)) 
sns.boxplot(x=df['math score'], ax=axes[0])
sns.boxplot(x=df['writing score'], ax=axes[1])
sns.boxplot(x=df['reading score'], ax=axes[2])
plt.show()
"""We have found outliers in all the three scores, whose count shall be taken and accordingly be removed"""
#Outliers found only on the left side of the boxplot
IQRm = df['math score'].quantile(0.75) - df['math score'].quantile(0.25)
lowerm = df['math score'].quantile(0.25) - 1.5 * IQRm
IQRw = df['writing score'].quantile(0.75) - df['writing score'].quantile(0.25)
lowerw = df['writing score'].quantile(0.25) - 1.5 * IQRw
IQRr = df['reading score'].quantile(0.75) - df['reading score'].quantile(0.25)
lowerr = df['reading score'].quantile(0.25) - 1.5 * IQRr
#Count of outliers all together that are to be removed
df[(df['math score']<lowerm) | (df['writing score']<lowerw) | (df['reading score']<lowerr)].count()

#12 is negligible among a 1000 rows dataset, so we can remove them
df = df[((df['math score']>=lowerm) & (df['writing score']>=lowerw) & (df['reading score']>=lowerr))]
df #Outliers removed
"""Once the outliers are removed, we can check the distribution of the scores in the dataset."""
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
sns.histplot(df['math score'], kde=True, ax=axes[0])
sns.histplot(df['writing score'], kde=True, ax=axes[1])
sns.histplot(df['reading score'], kde=True, ax=axes[2])
plt.show()
#Near Gaussion distribution, so we can proceed with encoding the categorical variables and scaling the numerical variables.
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
#Encoding the categorical variables
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])
df
"""We need to change the 'some college' to 'bachelor's degree' and 'some high school' to 'high school' in the parental level of education column"""
df['parental level of education'] = df['parental level of education'].replace({'some college': 'bachelor\'s degree', 'some high school': 'high school'})
df['parental level of education'].unique()
#Now we can encode the parental level of education column using the Label Encoder
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df
#OUR DATASET IS READY FOR MACHINE LEARNING MODELS
"""
There is something that we need to note before we proceed with the machine learning models.
We are using the categorical variables and data to predict the scores of the students and we are not using the scores to predict each other.
We can use different models to predict in consideration of the input:
1. If no score is given as input, we can use the categorical variables to predict the scores as a range of marks, this can have a wider range.
2. If one score is given as input, we can use the categorical variables and that one score to predict the missing score.
3. If two scores are given as input, we can use the categorical variables and those two scores to predict the missing score.
And if all three are given then there is no need to predict anything.
"""
"""This function is used to predict the scores of the students based on the categorical variables only.
   Through tests it has been seen that the scores predicted are really fitting to the actual scores but this being a completely categorical input to regression
    model, the range of marks is quite wide so it cannot be used to predict the exact scores.
"""
X = df.drop(['math score', 'writing score', 'reading score'], axis=1)
y = df[['math score', 'writing score', 'reading score']]
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict for X_test
preds = model.predict(X_test)
for i in range(len(preds)):
    print(preds[i], y_test.iloc[i].values) #ONLY FOR TESTING PURPOSES
    
"""Function to prepare the model data based on the input data provided by the user.
This function will check which scores are provided and prepare the input data accordingly."""    
def encode_input(input_data):
    """Encodes the input dictionary to match training encodings"""
    input_encoded = input_data.copy()
    enc_map = {
        'gender': {'female': 0, 'male': 1},
        'race/ethnicity': {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4},
        'lunch': {'free/reduced': 0, 'standard': 1},
        'test preparation course': {'none': 0, 'completed': 1},
        'parental level of education': {
            "associate's degree": 0,
            "bachelor's degree": 1,
            'high school': 2,
            "master's degree": 3,
            'some college': 1,  # mapped already
            'some high school': 2  # mapped already
        }
    }
    for col, mapping in enc_map.items():
        if col in input_encoded:
            input_encoded[col] = mapping.get(input_encoded[col], input_encoded[col])
    return input_encoded

"""Implementing this model to the required functions that predict scores based on input
FINAL USABLE FUNCTION CASE.1"""
def predict_cat(input_data):
    input_data = encode_input(input_data)
    cat_only_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cat_only_model.fit(X, y)
    input_df = pd.DataFrame([input_data])
    input_df = input_df.drop(['math score', 'writing score', 'reading score'], axis=1)
    prediction = cat_only_model.predict(input_df)
    return prediction[0]
#This function takes a dictionary of input data with categorical variables and returns the predicted scores as a list.
"""Now we also need to add the functionality to predict the missing scores based on the given input and categorical variables.
This function will take the input data with one or two scores and return the predicted missing score.
FINAL USABLE FUNCTION CASE.2"""
def predict_general(input_data):
    # Check how many scores are provided
    input_data = encode_input(input_data)
    score_count = sum(1 for score in input_data.values() if isinstance(score, (int, float)))
    if score_count == 0:
        return predict_cat(input_data)  # Use the categorical model if no scores are provided
    #What we can do now is to check which scores are missing and then create a model trained to predict that score based on the other scores and categorical variables.
    """We will first take the input data and check which columns are present, and then use df[input columns] to get the training data for the model.
    For example, in the input if the data for math score is given and rest aren't, we know what columns are filled so we use that as training dataset and the missing columns as prediction"""
    
    """
        df         : your full training dataset
        user_input : dict of inputs from web form, like {'gender': 'male', 'math score': 88}
        
        Returns:
            model_input : DataFrame row with same columns as training features
            X_train     : Features for training model
            Y_train     : Targets to predict
        """
    input_cols = list(input_data.keys())
        # Define target columns: columns in df not filled by user
    target_cols = [col for col in df.columns if col not in input_cols]
        # Prepare training data
    X_train = df[input_cols]
    Y_train = df[target_cols]
        # Prepare user input as single-row DataFrame
    input_df = pd.DataFrame([input_data])[input_cols]
    general_model = RandomForestRegressor(n_estimators=100, random_state=42)
    general_model.fit(X_train, Y_train)
    prediction = general_model.predict(input_df)
    return prediction[0]
