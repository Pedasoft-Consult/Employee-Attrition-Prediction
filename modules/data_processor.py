import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from io import StringIO

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_features = None
        self.categorical_features = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the data including proper encoding of target variable"""
        # Encode target variable first
        le = LabelEncoder()
        self.df['Attrition'] = le.fit_transform(self.df['Attrition'])
        self.label_encoders['Attrition'] = le

        # Identify numerical and categorical features
        self.numerical_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        # Remove target variable from features
        if 'Attrition' in self.numerical_features:
            self.numerical_features.remove('Attrition')

        # Store feature columns order
        self.feature_columns = self.numerical_features + self.categorical_features

        # Handle missing values
        for col in self.numerical_features:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        # Encode categorical variables
        for col in self.categorical_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        # Scale numerical features
        self.df[self.numerical_features] = self.scaler.fit_transform(self.df[self.numerical_features])

    def get_processed_data(self):
        """Get processed features and target for model training"""
        X = self.df[self.feature_columns]
        y = self.df['Attrition']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def get_numerical_features(self):
        """Get list of numerical feature names"""
        return self.numerical_features

    def get_categorical_features(self):
        """Get list of categorical feature names"""
        return self.categorical_features

    def get_job_roles(self):
        """Get list of unique job roles"""
        return self.label_encoders['JobRole'].classes_

    def get_data_info(self):
        """Get dataset information"""
        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def process_input_data(self, input_data):
        """Process input data for prediction with proper error handling"""
        try:
            processed_data = {}

            # Process numerical features
            for col in self.numerical_features:
                if col in input_data:
                    value = float(input_data[col])
                    processed_data[col] = (value - self.scaler.mean_[self.numerical_features.index(col)]) / \
                                            self.scaler.scale_[self.numerical_features.index(col)]
                else:
                    processed_data[col] = 0  # Default to mean (0 after scaling)

            # Process categorical features
            for col in self.categorical_features:
                if col in input_data:
                    try:
                        processed_data[col] = self.label_encoders[col].transform([str(input_data[col])])[0]
                    except ValueError as e:
                        raise ValueError(f"Invalid value for {col}: {input_data[col]}. Valid values are: {list(self.label_encoders[col].classes_)}")
                else:
                    processed_data[col] = 0  # Default to first category

            # Create DataFrame with correct column order
            return pd.DataFrame([processed_data], columns=self.feature_columns)

        except Exception as e:
            raise ValueError(f"Error processing input data: {str(e)}")

    def get_full_data(self):
        """Get all processed features and target for cross-validation"""
        X = self.df[self.feature_columns]
        y = self.df['Attrition']
        return X, y