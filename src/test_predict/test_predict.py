import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.predict_model import predict  # Import after updating the path

# Example vehicle features
sample_features = {
    'weight': 3000,         # weight
    'acceleration': 12.0,   # acceleration
    'displacement': 200.0,  # displacement
    'cylinders': 4,         # cylinders
    'model_year': 76,       # model year
    'horsepower': 90.0      # horsepower
}

# Call the function
result = predict(sample_features)

# Display the result
print("Estimated real fuel consumption (MPG):", result['consommation_reelle_mpg'])