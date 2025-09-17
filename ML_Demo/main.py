from sklearn.model_selection import train_test_split
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, get_features_target
from src.model_training import train_model
from src.evaluation import evaluate_model

# Load and preprocess data
df = load_data()
df, le_sex, le_embarked = preprocess_data(df)
X, y = get_features_target(df)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Terminal input for prediction
print("\n--- Titanic Survival Prediction ---")
pclass = int(input("Enter Pclass (1, 2, 3): "))
sex = input("Enter Sex (male/female): ")
age = float(input("Enter Age: "))
sibsp = int(input("Enter SibSp (siblings/spouses aboard): "))
parch = int(input("Enter Parch (parents/children aboard): "))
fare = float(input("Enter Fare: "))
embarked = input("Enter Embarked (C/Q/S): ")

# Encode inputs
sex_enc = le_sex.transform([sex])[0]
embarked_enc = le_embarked.transform([embarked])[0]

input_data = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Predict and display result
prediction = model.predict(input_data)
print("\nPrediction Result:")
print("Survived" if prediction[0] == 1 else "Did not survive")
