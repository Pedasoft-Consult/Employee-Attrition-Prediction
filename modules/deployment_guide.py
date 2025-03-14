import streamlit as st
from typing import Dict, List, Tuple

class DeploymentGuide:
    def __init__(self):
        self.checklist_items = {
            'data_validation': {
                'title': 'Data Validation',
                'items': [
                    ('data_quality', 'Data quality checks completed'),
                    ('feature_engineering', 'Feature engineering pipeline validated'),
                    ('data_preprocessing', 'Data preprocessing steps documented')
                ]
            },
            'model_validation': {
                'title': 'Model Validation',
                'items': [
                    ('cross_validation', 'Cross-validation performed'),
                    ('metrics_documented', 'Performance metrics documented'),
                    ('feature_importance', 'Feature importance analyzed')
                ]
            },
            'deployment_readiness': {
                'title': 'Deployment Readiness',
                'items': [
                    ('requirements_documented', 'Dependencies and requirements documented'),
                    ('error_handling', 'Error handling implemented'),
                    ('performance_tested', 'Performance testing completed')
                ]
            }
        }
        
    def validate_checklist_item(self, item_key: str, model_trainer) -> Tuple[bool, str]:
        """Validate individual checklist items"""
        if not model_trainer.is_trained():
            return False, "Model needs to be trained first"
            
        validations = {
            'cross_validation': lambda: 'CV Mean Accuracy' in model_trainer.get_latest_metrics(),
            'metrics_documented': lambda: all(metric in model_trainer.get_latest_metrics() 
                                           for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']),
            'feature_importance': lambda: model_trainer.get_feature_importance() is not None
        }
        
        if item_key in validations:
            try:
                is_valid = validations[item_key]()
                message = "Validated successfully" if is_valid else "Validation failed"
                return is_valid, message
            except Exception as e:
                return False, f"Error during validation: {str(e)}"
        
        return True, "Manual verification required"
    
    def render_checklist(self, model_trainer) -> Dict[str, bool]:
        """Render the deployment checklist in the Streamlit interface"""
        st.header("Deployment Checklist")
        
        checklist_status = {}
        
        for section_key, section in self.checklist_items.items():
            st.subheader(section['title'])
            for item_key, item_description in section['items']:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_checked = st.checkbox(
                        item_description,
                        key=f"checklist_{item_key}"
                    )
                    checklist_status[item_key] = is_checked
                
                with col2:
                    if st.button("Validate", key=f"validate_{item_key}"):
                        is_valid, message = self.validate_checklist_item(item_key, model_trainer)
                        if is_valid:
                            st.success("✓")
                        else:
                            st.error("✗")
                        st.toast(message)
                
                with col3:
                    if st.button("Info", key=f"info_{item_key}"):
                        self.show_item_info(item_key)
        
        return checklist_status
    
    def show_item_info(self, item_key: str):
        """Show detailed information about checklist items"""
        info_content = {
            'data_quality': """
            Ensure your dataset:
            - Has no missing values
            - Features are properly scaled
            - No unexpected outliers
            """,
            'cross_validation': """
            Cross-validation should:
            - Use k-fold validation (k=5)
            - Show consistent performance
            - Have acceptable variance
            """,
            'feature_importance': """
            Feature importance analysis should:
            - Identify key predictors
            - Remove irrelevant features
            - Document feature relationships
            """
        }
        
        if item_key in info_content:
            st.info(info_content[item_key])
