# From examples/predict.py
from brinias import make_prediction

print("--- MAKING A NEW PREDICTION ---")

# Create a dictionary with a new data point
# The keys MUST match the column names from your training CSV
new_data_point = {
   "timestamp": "2025-05-14",
   "Open": 2679.71,
   "High": 2725.99,
   "Low": 2547.26,
   "Close": 2609.74,
   "Volume": 830047.1122,
}

# Call the prediction function
prediction = make_prediction(
    input_data=new_data_point,
    model_dir="next_close" # Must match the output_dir from training
)

print("\n--- Prediction Result ---")
print(prediction)

# Example of how to use the result
if prediction.get('prediction_type') == 'classification':
    print(f"The predicted label is: {prediction.get('predicted_label')}")