import streamlit as st
import pandas as pd
from modules.data_processor import DataProcessor
from modules.model_trainer import ModelTrainer
from modules.visualization import Visualizer
from modules.deployment_guide import DeploymentGuide

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

@st.cache_data
def load_data():
    try:
        # Using the IBM HR Analytics dataset
        df = pd.read_csv('data/HR-Employee-Attrition.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("Employee Attrition Prediction System")

    # Load and process data
    df = load_data()
    if df is None:
        st.error("Failed to load the dataset. Please ensure the data file exists.")
        return

    # Initialize processors
    processor = DataProcessor(df)
    visualizer = Visualizer(df)

    # Initialize model trainer in session state
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer(processor)

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select a Page",
        ["Data Overview", "Exploratory Analysis", "Model Training", "Predictions", "Deployment Guide"]
    )

    if page == "Data Overview":
        st.header("Dataset Overview")
        st.write("Sample of the dataset:")
        st.dataframe(df.head())

        st.subheader("Dataset Statistics")
        st.write(df.describe())

        st.subheader("Data Info")
        buffer = processor.get_data_info()
        st.text(buffer)

    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")

        # Numerical distributions
        st.subheader("Numerical Feature Distributions")
        selected_num_feature = st.selectbox(
            "Select Numerical Feature",
            processor.get_numerical_features()
        )
        fig = visualizer.plot_numerical_distribution(selected_num_feature)
        st.plotly_chart(fig)

        # Categorical distributions
        st.subheader("Categorical Feature Analysis")
        selected_cat_feature = st.selectbox(
            "Select Categorical Feature",
            processor.get_categorical_features()
        )
        fig = visualizer.plot_categorical_distribution(selected_cat_feature)
        st.plotly_chart(fig)

        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig = visualizer.plot_correlation_matrix()
        st.plotly_chart(fig)

    elif page == "Model Training":
        st.header("Model Training and Evaluation")

        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression", "Decision Tree"]
        )

        if st.button("Train Model"):
            with st.spinner("Training model and performing cross-validation..."):
                try:
                    metrics, conf_matrix, feature_imp = st.session_state.model_trainer.train_model(model_type)

                    st.success("Model trained successfully!")

                    # Display metrics in two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Model Performance Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            'Value': [
                                metrics['Accuracy'],
                                metrics['Precision'],
                                metrics['Recall'],
                                metrics['F1 Score']
                            ]
                        })
                        st.table(metrics_df)

                    with col2:
                        st.subheader("Cross-Validation Results")
                        cv_metrics_df = pd.DataFrame({
                            'Metric': ['CV Mean Accuracy', 'CV Std Accuracy', 'CV Mean F1', 'CV Std F1'],
                            'Value': [
                                metrics['CV Mean Accuracy'],
                                metrics['CV Std Accuracy'],
                                metrics['CV Mean F1'],
                                metrics['CV Std F1']
                            ]
                        })
                        st.table(cv_metrics_df)

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    st.plotly_chart(visualizer.plot_confusion_matrix(conf_matrix))

                    # Feature Importance
                    st.subheader("Feature Importance Analysis")
                    st.plotly_chart(visualizer.plot_feature_importance(feature_imp))

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

    elif page == "Predictions":
        st.header("Make Predictions")

        if not st.session_state.model_trainer.is_trained():
            st.warning("Please train a model first before making predictions.")
            return

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6000)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

        with col2:
            job_role = st.selectbox("Job Role", processor.get_job_roles())
            overtime = st.selectbox("Overtime", ["Yes", "No"])

        if st.button("Predict"):
            try:
                input_data = {
                    'Age': age,
                    'MonthlyIncome': monthly_income,
                    'YearsAtCompany': years_at_company,
                    'JobRole': job_role,
                    'OverTime': overtime
                }

                prediction, probability = st.session_state.model_trainer.predict(input_data)

                st.subheader("Prediction Result")
                st.write(f"Attrition Prediction: {'Yes' if prediction else 'No'}")
                st.write(f"Probability of Attrition: {probability:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif page == "Deployment Guide":
        st.header("Model Deployment Guide")

        if not st.session_state.model_trainer.is_trained():
            st.warning("Please train a model first before accessing the deployment guide.")
            return

        deployment_guide = DeploymentGuide()
        checklist_status = deployment_guide.render_checklist(st.session_state.model_trainer)

        # Show deployment readiness status
        st.subheader("Deployment Readiness Status")
        total_items = len(checklist_status)
        completed_items = sum(checklist_status.values())

        progress = completed_items / total_items
        st.progress(progress)
        st.write(f"Completed: {completed_items}/{total_items} items")

        if progress == 1.0:
            st.success("🎉 Your model is ready for deployment!")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Model"):
                    try:
                        export_dir = st.session_state.model_trainer.export_model()
                        st.success(f"Model exported successfully to {export_dir}")
                        st.info("The exported package includes:\n"
                                "- Trained model\n"
                                "- Preprocessing components\n"
                                "- Model performance metrics\n"
                                "- Requirements file")
                    except Exception as e:
                        st.error(f"Error exporting model: {str(e)}")

            with col2:
                if st.button("Deploy Model"):
                    suggest_deploy()
        else:
            st.info("Complete all checklist items to proceed with deployment")


def suggest_deploy():
    st.write("## Model Deployment Options")

    st.markdown("""
    ### 1. Replit Deployment (Recommended)
    - **Advantages**: 
        - Seamless integration with your current environment
        - Automatic scaling and HTTPS
        - Built-in monitoring
    - **Steps**:
        1. Click 'Deploy' in the Replit interface
        2. Choose deployment settings
        3. Your model will be accessible via API endpoints

    ### 2. Alternative Cloud Platforms

    #### AWS SageMaker
    - Perfect for production ML workloads
    - Offers automatic scaling
    - Integrated monitoring
    - **Requirements**:
        - AWS account
        - Exported model (already provided)
        - API endpoint configuration

    #### Google Cloud AI Platform
    - Suitable for TensorFlow/scikit-learn models
    - Offers serverless deployment
    - **Requirements**:
        - Google Cloud account
        - Exported model format
        - Service account setup

    ### Security Considerations
    - Ensure API authentication
    - Implement rate limiting
    - Monitor model performance
    - Regular updates and maintenance

    ### Next Steps
    1. Download the exported model package
    2. Follow platform-specific deployment guide
    3. Set up monitoring and logging
    4. Implement proper security measures
    """)

    # Add deployment action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Deploy on Replit"):
            suggest_deploy()
    with col2:
        if st.button("Download Deployment Guide"):
            st.markdown("""
            A comprehensive deployment guide has been included in the
            exported model package under 'deployment_guide.md'
            """)


if __name__ == "__main__":
    main()