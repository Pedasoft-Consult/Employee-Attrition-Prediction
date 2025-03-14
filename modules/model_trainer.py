import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

class ModelTrainer:
    def __init__(self, processor):
        self.processor = processor
        self.model = None
        self.feature_names = None
        self.model_type = None

    def train_model(self, model_type):
        """Train selected model and return evaluation metrics"""
        try:
            # Get processed data
            X_train, X_test, y_train, y_test = self.processor.get_processed_data()
            self.feature_names = X_train.columns.tolist()
            self.model_type = model_type

            # Select and configure model
            if model_type == "Random Forest":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "Logistic Regression":
                self.model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
            else:  # Decision Tree
                self.model = DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )

            # Train model
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            metrics = {
                'Accuracy': round(accuracy_score(y_test, y_pred), 3),
                'Precision': round(precision_score(y_test, y_pred), 3),
                'Recall': round(recall_score(y_test, y_pred), 3),
                'F1 Score': round(f1_score(y_test, y_pred), 3)
            }

            # Get confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                importances = np.abs(self.model.coef_[0])

            feature_imp = dict(sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            ))

            # Perform cross-validation
            cv_scores = self.cross_validate()
            metrics['CV Mean Accuracy'] = round(cv_scores['accuracy'].mean(), 3)
            metrics['CV Std Accuracy'] = round(cv_scores['accuracy'].std(), 3)
            metrics['CV Mean F1'] = round(cv_scores['f1'].mean(), 3)
            metrics['CV Std F1'] = round(cv_scores['f1'].std(), 3)

            return metrics, conf_matrix, feature_imp

        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")

    def cross_validate(self, k=5):
        """Perform k-fold cross-validation"""
        X, y = self.processor.get_full_data()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        accuracy_scores = []
        f1_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)

            accuracy_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred))

        return {
            'accuracy': np.array(accuracy_scores),
            'f1': np.array(f1_scores)
        }

    def is_trained(self):
        """Check if model has been trained"""
        return self.model is not None and self.model_type is not None

    def predict(self, input_data):
        """Make prediction for input data"""
        if not self.is_trained():
            raise RuntimeError("Model not trained. Please train a model first.")

        try:
            # Process input data
            processed_input = self.processor.process_input_data(input_data)

            # Verify input data has correct features
            if not all(col in processed_input.columns for col in self.feature_names):
                missing = set(self.feature_names) - set(processed_input.columns)
                raise ValueError(f"Missing features in input data: {missing}")

            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            probability = self.model.predict_proba(processed_input)[0][1]

            return prediction, probability

        except Exception as e:
            raise RuntimeError(f"Error making prediction: {str(e)}")

    def get_latest_metrics(self):
        """Get the latest training metrics"""
        if not self.is_trained():
            return {}
        try:
            X_train, X_test, y_train, y_test = self.processor.get_processed_data()
            y_pred = self.model.predict(X_test)

            return {
                'Accuracy': round(accuracy_score(y_test, y_pred), 3),
                'Precision': round(precision_score(y_test, y_pred), 3),
                'Recall': round(recall_score(y_test, y_pred), 3),
                'F1 Score': round(f1_score(y_test, y_pred), 3)
            }
        except Exception:
            return {}

    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained():
            return None

        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                importances = np.abs(self.model.coef_[0])

            return dict(sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            ))
        except Exception:
            return None

    def export_model(self, export_path='exported_model'):
        """Export trained model and related artifacts"""
        try:
            import joblib
            import os
            from datetime import datetime

            if not self.is_trained():
                raise RuntimeError("Model not trained. Please train the model first.")

            # Create export directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"{export_path}_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)

            # Export model
            model_path = os.path.join(export_dir, 'model.joblib')
            joblib.dump(self.model, model_path)

            # Export scaler and encoders
            preprocessing_path = os.path.join(export_dir, 'preprocessor.joblib')
            preprocessing = {
                'scaler': self.processor.scaler,
                'label_encoders': self.processor.label_encoders,
                'numerical_features': self.processor.numerical_features,
                'categorical_features': self.processor.categorical_features,
                'feature_columns': self.processor.feature_columns
            }
            joblib.dump(preprocessing, preprocessing_path)

            # Export model info
            model_info = {
                'model_type': self.model_type,
                'metrics': self.get_latest_metrics(),
                'feature_importance': self.get_feature_importance()
            }

            info_path = os.path.join(export_dir, 'model_info.txt')
            with open(info_path, 'w') as f:
                f.write("Model Information:\n")
                f.write(f"Model Type: {model_info['model_type']}\n\n")
                f.write("Performance Metrics:\n")
                for metric, value in model_info['metrics'].items():
                    f.write(f"{metric}: {value}\n")
                f.write("\nFeature Importance:\n")
                for feature, importance in model_info['feature_importance'].items():
                    f.write(f"{feature}: {importance:.4f}\n")

            # Create deployment guide
            guide_path = os.path.join(export_dir, 'deployment_guide.md')
            with open(guide_path, 'w') as f:
                f.write("""# Employee Attrition Model Deployment Guide

## Overview
This guide provides detailed instructions for deploying the Employee Attrition Prediction model in various environments.

## Package Contents
- `model.joblib`: Trained machine learning model
- `preprocessor.joblib`: Data preprocessing components
- `model_info.txt`: Model performance metrics and features
- `requirements.txt`: Required Python packages

## Deployment Options

### 1. Replit Deployment (Recommended)

#### Prerequisites
- Replit account
- Basic Python knowledge

#### Steps
1. Create a new Replit project
2. Upload the exported model package
3. Install requirements: `pip install -r requirements.txt`
4. Create an API endpoint using Flask/FastAPI
5. Deploy using Replit's deployment feature

#### Example API Implementation
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process input using preprocessor
    processed_data = preprocess_input(data)
    # Make prediction
    prediction = model.predict(processed_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. AWS SageMaker Deployment

#### Prerequisites
- AWS account
- AWS CLI configured
- Basic AWS knowledge

#### Steps
1. Package model for SageMaker
2. Create SageMaker endpoint
3. Configure API Gateway
4. Set up monitoring

### 3. Google Cloud AI Platform

#### Prerequisites
- Google Cloud account
- gcloud CLI configured
- Service account with required permissions

#### Steps
1. Package model for AI Platform
2. Create new model version
3. Deploy endpoint
4. Set up monitoring

## Security Best Practices
1. Implement API authentication
2. Use HTTPS endpoints
3. Rate limit API calls
4. Monitor for unusual patterns
5. Regular security updates

## Monitoring
- Track prediction accuracy
- Monitor system resources
- Set up alerts for errors
- Log all predictions

## Troubleshooting
- Check input data format
- Verify preprocessing steps
- Monitor error logs
- Test endpoint connectivity

## Support
For technical support or questions about deployment:
1. Check documentation
2. Review error logs
3. Contact system administrator

""")

            # Create requirements.txt
            requirements_path = os.path.join(export_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write("numpy\npandas\nscikit-learn\njoblib\nflask\ngunicorn\n")

            return export_dir

        except Exception as e:
            raise RuntimeError(f"Error exporting model: {str(e)}")