
import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json

# Load the saved model and preprocessing artifacts
model = load_model(r'D:\DEEP LEARNING\01. Campusx\3.Customer Churn Prediction\customer_churn_model.h5')
scaler = joblib.load('scaler.save')

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
def predict_churn(
    CreditScore,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    HasCrCard,
    IsActiveMember,
    EstimatedSalary,
    Geography,
    Gender
):
    """Make a churn prediction based on user input"""
    # Create a dictionary from inputs
    input_data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography_France': 1 if Geography == "France" else 0,
        'Geography_Germany': 1 if Geography == "Germany" else 0,
        'Geography_Spain': 1 if Geography == "Spain" else 0,
        'Gender_Male': 1 if Gender == "Male" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale the features
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction_prob = model.predict(scaled_input)[0][0]
    prediction = "Churn" if prediction_prob > 0.5 else "Not Churn"
    confidence = "High" if abs(prediction_prob - 0.5) > 0.3 else "Medium" if abs(prediction_prob - 0.5) > 0.1 else "Low"
    
    # Return three separate values instead of a dictionary
    return prediction, float(prediction_prob), confidence

# Create Gradio interface
with gr.Blocks(title="Customer Churn Predictor") as app:
    gr.Markdown("# 🏦 Customer Churn Prediction")
    gr.Markdown("Predict whether a customer will leave the bank")
    
    with gr.Row():
        with gr.Column():
            CreditScore = gr.Slider(300, 850, value=650, label="Credit Score")
            Age = gr.Slider(18, 100, value=35, label="Age")
            Tenure = gr.Slider(0, 15, value=5, label="Tenure (years)")
            Balance = gr.Number(value=100000, label="Account Balance")
            NumOfProducts = gr.Slider(1, 4, value=2, label="Number of Products")
            
        with gr.Column():
            HasCrCard = gr.Radio([1, 0], label="Has Credit Card?", info="1=Yes, 0=No")
            IsActiveMember = gr.Radio([1, 0], label="Is Active Member?", info="1=Yes, 0=No")
            EstimatedSalary = gr.Number(value=150000, label="Estimated Salary ($)")
            Geography = gr.Dropdown(["France", "Germany", "Spain"], value="France", label="Country")
            Gender = gr.Radio(["Male", "Female"], label="Gender")
    
    submit_btn = gr.Button("Predict Churn Risk", variant="primary")
    
    # Output components
    with gr.Accordion("Prediction Results", open=True):
        prediction_output = gr.Label(label="Prediction")
        probability_output = gr.Number(label="Probability", precision=3)
        confidence_output = gr.Textbox(label="Confidence Level")
    
    submit_btn.click(
        fn=predict_churn,
        inputs=[
            CreditScore,
            Age,
            Tenure,
            Balance,
            NumOfProducts,
            HasCrCard,
            IsActiveMember,
            EstimatedSalary,
            Geography,
            Gender
        ],
        outputs=[prediction_output, probability_output, confidence_output]
    )
    
    gr.Examples(
        examples=[
            [650, 35, 5, 100000, 2, 1, 1, 150000, "France", "Male"],
            [580, 42, 3, 0, 1, 1, 0, 80000, "Germany", "Female"],
            [720, 28, 2, 150000, 3, 0, 1, 200000, "Spain", "Male"]
        ],
        inputs=[
            CreditScore,
            Age,
            Tenure,
            Balance,
            NumOfProducts,
            HasCrCard,
            IsActiveMember,
            EstimatedSalary,
            Geography,
            Gender
        ]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)  # Set share=True to get a public link