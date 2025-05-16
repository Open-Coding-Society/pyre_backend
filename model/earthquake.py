import numpy as np
import pandas as pd
import joblib  # For loading the model
from sklearn.ensemble import RandomForestRegressor
from __init__ import app, db

class EarthquakeModel:
    _instance = None

    @staticmethod
    def get_instance():
        if EarthquakeModel._instance is None:
            EarthquakeModel()
        return EarthquakeModel._instance

    def __init__(self):
        if EarthquakeModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EarthquakeModel._instance = self
            # Initialize the model
            self.model = self._load_model()
            self.time_map = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
            self.plate_type_map = {'convergent': 0, 'divergent': 1, 'transform': 2, 'intraplate': 3}
            self.soil_type_map = {'rock': 0, 'stiff_soil': 1, 'soft_soil': 2, 'alluvium': 3}

    def _load_model(self):
        # Load the pre-trained model from a file
        try:
            model = joblib.load('earthquake_model.pkl')  # Replace with your model file path
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            raise Exception("Model file not found. Please ensure 'earthquake_model.pkl' exists.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}")

    def predict(self, earthquake_data):
        # Prepare the input data for prediction
        try:
            features = np.array([
                earthquake_data['latitude'],
                earthquake_data['longitude'],
                earthquake_data['depth'],
                self.time_map[earthquake_data['time_of_day']],
                earthquake_data['previous_magnitude'],
                earthquake_data['distance_to_fault'],
                self.plate_type_map[earthquake_data['plate_boundary_type']],
                self.soil_type_map[earthquake_data['soil_type']]
            ]).reshape(1, -1)

            # Use the loaded model to make a prediction
            predicted_magnitude = self.model.predict(features)[0]
            return {"predicted_magnitude": predicted_magnitude}
        except KeyError as e:
            return {"error": f"Missing or invalid field: {e}"}
        except Exception as e:
            return {"error": f"An error occurred during prediction: {e}"}