import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_numerical_distribution(self, feature):
        fig = px.histogram(
            self.df,
            x=feature,
            color='Attrition',
            marginal='box',
            title=f'Distribution of {feature}'
        )
        return fig
        
    def plot_categorical_distribution(self, feature):
        counts = pd.crosstab(self.df[feature], self.df['Attrition'])
        fig = px.bar(
            counts,
            barmode='group',
            title=f'{feature} vs Attrition'
        )
        return fig
        
    def plot_correlation_matrix(self):
        corr = self.df.select_dtypes(include=['int64', 'float64']).corr()
        fig = px.imshow(
            corr,
            title='Correlation Matrix',
            aspect='auto'
        )
        return fig
        
    def plot_confusion_matrix(self, conf_matrix):
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['No', 'Yes'],
            y=['No', 'Yes'],
            title='Confusion Matrix'
        )
        return fig
        
    def plot_feature_importance(self, feature_imp):
        """Enhanced feature importance visualization with detailed analysis"""
        feature_imp_df = pd.DataFrame({
            'Feature': list(feature_imp.keys()),
            'Importance': list(feature_imp.values())
        }).sort_values('Importance', ascending=True)

        # Create detailed bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_imp_df['Importance'],
            y=feature_imp_df['Feature'],
            orientation='h',
            marker=dict(
                color=feature_imp_df['Importance'],
                colorscale='Viridis'
            )
        ))

        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(feature_imp) * 20),
            showlegend=False
        )

        return fig