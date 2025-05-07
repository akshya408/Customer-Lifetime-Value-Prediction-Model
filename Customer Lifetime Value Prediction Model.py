import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

MODEL_FILENAME = 'clv_model.joblib'

def generate_synthetic_data(num_customers=1000, random_state=42):
    np.random.seed(random_state)
    recency = np.random.exponential(scale=30, size=num_customers)
    frequency = np.random.poisson(lam=5, size=num_customers) + 1
    monetary = np.random.gamma(shape=2, scale=50, size=num_customers)
    clv = (frequency * monetary) / (recency + 1) + np.random.normal(0, 10, size=num_customers)
    clv = np.maximum(clv, 0)
    data = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'clv': clv
    })
    return data

def train_clv_model(data):
    X = data[['recency', 'frequency', 'monetary']]
    y = data['clv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

MODEL_FILENAME = 'clv_model.joblib'

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        return model, None, None
    else:
        data = generate_synthetic_data()
        model, mse, r2 = train_clv_model(data)
        joblib.dump(model, MODEL_FILENAME)
        return model, mse, r2

def plot_feature_importance(model):
    importances = model.feature_importances_
    features = ['recency', 'frequency', 'monetary']
    fig = go.Figure([go.Bar(y=features, x=importances, orientation='h', marker_color='indianred', text=importances, textposition='auto')])
    fig.update_layout(title='Feature Importance', xaxis_title='Importance', yaxis_title='Features')
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Customer Lifetime Value Prediction")
    st.write("Upload your customer data CSV file with columns: recency, frequency, monetary.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    model, mse, r2 = load_or_train_model()
    if mse is not None and r2 is not None:
        st.write(f"Model trained with Mean Squared Error: {mse:.2f}, R2 Score: {r2:.2f}")
    else:
        st.write("Loaded existing model.")
    
    # Load default dataset if no file uploaded
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = None
    
    if data is not None:
        st.write("Data preview:")
        # Limit preview to first 100 rows for performance
        preview_df = data.head(100)
        def style_df(df):
            return df.style.background_gradient(cmap='Blues').format(precision=2)
        st.dataframe(style_df(preview_df))
        
        st.write("Columns in dataset:")
        st.write(data.columns.tolist())
        
        st.write("Summary statistics:")
        st.write(data.describe())
        
        # Add model explanation or description here
        st.subheader("Model Explanation")
        # Dynamically generate description based on selected columns
        description_lines = [
            "This model is a Random Forest Regressor trained on synthetic customer data.",
            "It predicts Customer Lifetime Value (CLV) based on the following features:"
        ]
        selected_features = []
        # Use try-except to handle cases where recency_col etc. are not yet defined
        try:
            if recency_col in data.columns:
                selected_features.append(f"- {recency_col}: How recently a customer made a purchase.")
            if frequency_col in data.columns:
                selected_features.append(f"- {frequency_col}: How often a customer makes purchases.")
            if monetary_col in data.columns:
                selected_features.append(f"- {monetary_col}: How much money a customer spends.")
        except NameError:
            # Variables not defined yet, skip adding features
            pass
        if selected_features:
            description_lines.extend(selected_features)
        else:
            description_lines.append("No recognized features found in the dataset.")
        description_lines.append("\nThe model uses these features to estimate the expected value a customer will bring over time.")
        st.write("\n".join(description_lines))
        
        st.write("Select the columns corresponding to the features:")
        recency_col = st.selectbox("Recency column", options=data.columns)
        frequency_col = st.selectbox("Frequency column", options=data.columns)
        monetary_col = st.selectbox("Monetary column", options=data.columns)
        
        if st.button("Generate Report"):
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress
            for percent_complete in range(0, 101, 10):
                # Removed time.sleep to avoid blocking UI and lag
                progress_bar.progress(percent_complete)
                elapsed = time.time() - start_time
                status_text.text(f"Generating visuals... Elapsed time: {elapsed:.1f} seconds")
            
            # Predict CLV
            try:
                selected_cols = [recency_col, frequency_col, monetary_col]
                if not all(col in data.columns for col in selected_cols):
                    missing = [col for col in selected_cols if col not in data.columns]
                    st.error(f"Missing required columns for prediction: {missing}")
                else:
                    features = data[selected_cols].copy()
                    features.columns = ['recency', 'frequency', 'monetary']  # rename for model

                    # Preprocess features to convert ranges like '5L-10L' to numeric midpoints
                    import re
                    def convert_range_to_midpoint(val):
                        if isinstance(val, str):
                            match = re.match(r'(\d+(?:\.\d+)?)L-(\d+(?:\.\d+)?)L', val)
                            if match:
                                low = float(match.group(1))
                                high = float(match.group(2))
                                return (low + high) / 2
                            else:
                                try:
                                    return float(val)
                                except:
                                    return np.nan
                        else:
                            return val

                    for col in features.columns:
                        features[col] = features[col].apply(convert_range_to_midpoint)
                        features[col] = pd.to_numeric(features[col], errors='coerce')

                    # Drop rows with NaN after conversion
                    features = features.dropna()

                    predictions = model.predict(features)
                    data.loc[features.index, 'predicted_clv'] = predictions
                    
                    st.write("Predicted Customer Lifetime Value:")
                    st.dataframe(data[['predicted_clv']].head())
                    
                # Plot predicted CLV distribution
                st.subheader("Predicted CLV Distribution")
                sorted_preds = np.sort(predictions)
                fig = px.line(x=np.arange(len(sorted_preds)), y=sorted_preds, title='Predicted CLV Line Chart', labels={'x': 'Index', 'y': 'Predicted CLV'})
                fig.update_traces(line=dict(color='skyblue'), mode='lines+markers', text=[f'{v:.2f}' for v in sorted_preds], textposition='top center')
                fig.update_layout(xaxis_title='Index', yaxis_title='Predicted CLV')
                st.plotly_chart(fig, use_container_width=True, key="clv_line_chart")
                
                # Plot feature importance
                st.subheader("Feature Importance")
                plot_feature_importance(model)
                
                elapsed = time.time() - start_time
                status_text.text(f"Report generated in {elapsed:.1f} seconds.")
                progress_bar.progress(100)
            except Exception as e:
                st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
