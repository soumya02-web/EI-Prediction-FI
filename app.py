import gradio as gr
import joblib
import numpy as np

# Load saved models and transformers
model = joblib.load("best_rf_model.joblib")
scaler = joblib.load("scaler.joblib")
poly = joblib.load("poly.joblib")

# Define prediction function
def predict_eq(cl, wm, rt):
    try:
        rt_inv = 1 / (rt + 1e-6)
    except ZeroDivisionError:
        return "Invalid RT (cannot be zero)"
    
    X_input = np.array([[cl, wm, rt_inv]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)
    pred = model.predict(X_scaled)[0]
    return round(pred, 2)

# Gradio interface
interface = gr.Interface(
    fn=predict_eq,
    inputs=[
        gr.Number(label="Cognitive Load (cl)"),
        gr.Number(label="Working Memory (wm)"),
        gr.Number(label="Reaction Time (rt)")
    ],
    outputs=gr.Number(label="Predicted Emotional Intelligence (eq)"),
    title="Emotional Intelligence Predictor",
    description="Enter cognitive inputs to predict EQ"
)

if __name__ == "__main__":
    interface.launch()
