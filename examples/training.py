# From examples/training.py
from brinias import train_model

print("--- STARTING MODEL TRAINING ---")

# Define the training parameters
train_model(
    csv_path="dataeth.csv",
    target_column="next_close",
    output_dir="next_close",
    generations=120,   # Use a small number for a quick test
    pop_size=100,
    show_plot=True    # Display a plot of actual vs. predicted after training
)

print("--- TRAINING COMPLETE ---")

