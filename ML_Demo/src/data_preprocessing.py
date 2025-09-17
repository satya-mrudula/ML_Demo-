import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path='data/train.csv'):
    """Load Titanic dataset."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """Fill missing values and encode categorical features."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    return df, le_sex, le_embarked

def get_features_target(df):
    """Select features and target variable."""
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']
    return X, y
