📊 Student Score Predictor

This project predicts student exam scores using demographic and other categorical data. It is flexible in handling partial input — users can provide any combination of scores (math, reading, writing), and the model dynamically predicts the missing ones.

📝 Description

This project predicts student academic performance based on available information such as demographic and educational background. It is designed to work even if the user provides only partial score data (e.g., only math score, or none at all). The model intelligently adapts based on the available inputs to estimate missing exam scores.

The goal is to simulate a realistic academic advisor tool, where full data may not always be available, and decisions must be made from what's provided.

⚙️ Working

Input Collection

User inputs are taken for:

5 categorical features: gender, race/ethnicity, parental level of education, lunch, test preparation course

0, 1, or 2 of the 3 exam scores: math score, reading score, writing score

Dynamic Model Selection

Based on which score(s) are missing:

If all 3 scores are missing → train models to predict all 3.

If only 1 or 2 are missing → train models to predict just those.

The model uses only the available columns for training (X) and predicts the missing column(s) (y).

Encoding and Preprocessing

Categorical inputs are encoded using LabelEncoder or OrdinalEncoder.

The system automatically detects the correct column combination for training.

Model Training

A Random Forest Regressor is trained dynamically based on selected input features.

The more scores the user provides, the better the accuracy.

Score Prediction

The model predicts the missing score(s) and returns them.

In future versions, this could include confidence intervals or ranges instead of just point estimates.

🔄 Pipeline Diagram

User Input (Categorical + Optional Scores)
          ↓
   Identify Given Columns
          ↓
   Select Features (X) and Targets (y)
          ↓
   Encode Categorical Data
          ↓
  Train Random Forest Regressor
          ↓
   Predict Missing Scores
          ↓
       Output to User

❓ Why This Approach

Real-world data often has missing or partial information — this solution reflects that.

Allows flexibility in usage: full prediction, partial completion, or estimation.

Random Forest handles tabular data well and is robust to noise and categorical encodings.

Easily extensible to add:

Confidence intervals

Web interfaces

Continuous learning via user feedback

📄 Suitable for Resume / Project Reports

Developed a dynamic student performance prediction system using Random Forests that adapts to partially available data. Designed to predict one or more exam scores (math, reading, writing) using categorical features such as gender, parental education, and more. Implemented dynamic column selection, encoding pipelines, and model training to make accurate score estimates under uncertain input conditions.
