import numpy as np
import pandas as pd
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
            self.time_map = {'morning':0, 'afternoon':1, 'evening':2, 'night':3}
            self.plate_type_map = {'convergent':0, 'divergent':1, 'transform':2, 'intraplate':3}
            self.soil_type_map = {'rock':0, 'stiff_soil':1, 'soft_soil':2, 'alluvium':3}

    def _load_model(self):
        # In a real implementation, you would load a pre-trained model from disk
        # For this example, we'll create a simple model
        try:
            # Create synthetic earthquake data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic features
            latitudes = np.random.uniform(25, 50, n_samples)
            longitudes = np.random.uniform(-125, -65, n_samples)
            depths = np.random.exponential(10, n_samples)
            time_of_day = np.random.randint(0, 4, n_samples)
            previous_magnitudes = np.random.uniform(0, 7, n_samples)
            distance_to_fault = np.random.exponential(50, n_samples)
            plate_boundary_types = np.random.randint(0, 4, n_samples)
            soil_types = np.random.randint(0, 4, n_samples)
            
            # Create synthetic magnitudes with some correlation to features
            # Greater depth typically results in less surface magnitude
            # Closer to fault typically means stronger magnitude
            magnitudes = (
                0.5 * previous_magnitudes + 
                0.3 * np.random.normal(0, 1, n_samples) -
                0.01 * depths +
                0.1 * (100 - np.minimum(distance_to_fault, 100)) / 100 +
                0.1 * plate_boundary_types
            )
            magnitudes = np.clip(magnitudes, 1.0, 9.0)  # Limit to Richter scale range
            
            # Create DataFrame
            df = pd.DataFrame({
                'latitude': latitudes,
                'longitude': longitudes,
                'depth': depths,
                'time_of_day': time_of_day,
                'previous_magnitude': previous_magnitudes,
                'distance_to_fault': distance_to_fault,
                'plate_boundary_type': plate_boundary_types,
                'soil_type': soil_types,
                'magnitude': magnitudes
            })
            
            X = df.drop('magnitude', axis=1)
            y = df['magnitude']
            
            # Train a RandomForest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)  # Train on all data for deployment
            
            return rf
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def estimate_seismic_intensity(self, previous_magnitude, depth, distance_to_fault):
        """Estimate seismic intensity based on previous magnitude, depth, and distance to fault"""
        # Simple estimation formula (in reality, would be much more complex)
        return max(0, 1.2 * previous_magnitude - 0.02 * depth - 0.01 * distance_to_fault + 0.5)

    def estimate_ground_acceleration(self, seismic_intensity, distance_to_fault):
        """Estimate peak ground acceleration"""
        # Simple attenuation relationship
        return max(0, seismic_intensity * np.exp(-0.005 * distance_to_fault))

    def estimate_liquefaction_potential(self, latitude, longitude):
        """Estimate liquefaction potential based on location"""
        # In reality, this would use geological data about the location
        # For this example, we'll use a simple synthetic approach
        return max(0, min(1, 0.3 + 0.02 * np.sin(latitude) + 0.03 * np.cos(longitude)))

    def predict(self, earthquake_data):
        """
        Predict the magnitude based on input earthquake data

        Args:
            earthquake_data (dict): Dictionary containing input parameters

        Returns:
            dict: Dictionary with prediction results
        """
        try:
            # Process and prepare the input data
            latitude = earthquake_data.get('latitude', 0)
            longitude = earthquake_data.get('longitude', 0)
            depth = earthquake_data.get('depth', 0)
            time_of_day = earthquake_data.get('time_of_day', 'morning').lower()
            previous_magnitude = earthquake_data.get('previous_magnitude', 0)
            distance_to_fault = earthquake_data.get('distance_to_fault', 0)
            plate_boundary_type = earthquake_data.get('plate_boundary_type', 'transform').lower()
            soil_type = earthquake_data.get('soil_type', 'rock').lower()

            # Calculate seismic factors if not provided
            seismic_intensity = earthquake_data.get('seismic_intensity', 
                                                 self.estimate_seismic_intensity(previous_magnitude, depth, distance_to_fault))
            ground_acceleration = earthquake_data.get('ground_acceleration', 
                                                   self.estimate_ground_acceleration(seismic_intensity, distance_to_fault))
            liquefaction_potential = earthquake_data.get('liquefaction_potential', 
                                                    self.estimate_liquefaction_potential(latitude, longitude))

            # Convert categorical variables to numeric codes
            time_code = self.time_map.get(time_of_day, 0)
            plate_code = self.plate_type_map.get(plate_boundary_type, 0)
            soil_code = self.soil_type_map.get(soil_type, 0)

            # Create input vector
            input_vector = pd.DataFrame([{
                'latitude': latitude,
                'longitude': longitude,
                'depth': depth,
                'time_of_day': time_code,
                'previous_magnitude': previous_magnitude,
                'distance_to_fault': distance_to_fault,
                'plate_boundary_type': plate_code,
                'soil_type': soil_code
            }])

            # Make prediction
            predicted_magnitude = self.model.predict(input_vector)[0]

            return {
                "predicted_magnitude": float(predicted_magnitude),
                "risk_factors": {
                    "seismic_intensity": float(seismic_intensity),
                    "ground_acceleration": float(ground_acceleration),
                    "liquefaction_potential": float(liquefaction_potential)
                }
            }
        except Exception as e:
            return {"error": str(e)}


class Earthquake(db.Model):
    """
    Earthquake Model

    The Earthquake class represents an earthquake record with geological and location data.

    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the record.
        latitude (db.Column): A float representing the latitude coordinate of the earthquake epicenter.
        longitude (db.Column): A float representing the longitude coordinate of the earthquake epicenter.
        depth (db.Column): A float representing the depth of the earthquake in kilometers.
        time_of_day (db.Column): A string representing the time of day when the earthquake occurred.
        previous_magnitude (db.Column): A float representing the previous magnitude in the same region.
        distance_to_fault (db.Column): A float representing the distance to the nearest fault line in kilometers.
        plate_boundary_type (db.Column): A string representing the type of plate boundary.
        soil_type (db.Column): A string representing the predominant soil type in the area.
        magnitude (db.Column): A float representing the magnitude of the earthquake on the Richter scale.
    """
    __tablename__ = 'earthquakes'

    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    time_of_day = db.Column(db.String(10), nullable=False)
    previous_magnitude = db.Column(db.Float, nullable=False)
    distance_to_fault = db.Column(db.Float, nullable=False)
    plate_boundary_type = db.Column(db.String(20), nullable=False)
    soil_type = db.Column(db.String(20), nullable=False)
    magnitude = db.Column(db.Float, nullable=False)

    def __init__(self, latitude, longitude, depth, time_of_day, previous_magnitude, 
                 distance_to_fault, plate_boundary_type, soil_type, magnitude=0.0):
        """
        Constructor, initializes an Earthquake object.

        Args:
            latitude (float): The latitude coordinate of the earthquake epicenter.
            longitude (float): The longitude coordinate of the earthquake epicenter.
            depth (float): The depth of the earthquake in kilometers.
            time_of_day (str): The time of day when the earthquake occurred ('morning', 'afternoon', 'evening', 'night').
            previous_magnitude (float): The previous magnitude in the same region.
            distance_to_fault (float): The distance to the nearest fault line in kilometers.
            plate_boundary_type (str): The type of plate boundary ('convergent', 'divergent', 'transform', 'intraplate').
            soil_type (str): The predominant soil type in the area ('rock', 'stiff_soil', 'soft_soil', 'alluvium').
            magnitude (float, optional): The magnitude of the earthquake on the Richter scale. Defaults to 0.0.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time_of_day = time_of_day
        self.previous_magnitude = previous_magnitude
        self.distance_to_fault = distance_to_fault
        self.plate_boundary_type = plate_boundary_type
        self.soil_type = soil_type
        self.magnitude = magnitude

    def __repr__(self):
        """
        The __repr__ method is a special method used to represent the object in a string format.
        Called by the repr() built-in function.

        Returns:
            str: A text representation of how to create the object.
        """
        return f"Earthquake(id={self.id}, latitude={self.latitude}, longitude={self.longitude}, depth={self.depth}, " \
               f"time_of_day='{self.time_of_day}', previous_magnitude={self.previous_magnitude}, " \
               f"distance_to_fault={self.distance_to_fault}, plate_boundary_type='{self.plate_boundary_type}', " \
               f"soil_type='{self.soil_type}', magnitude={self.magnitude})"

    def create(self):
        """
        The create method adds the object to the database and commits the transaction.

        Uses:
            The db ORM methods to add and commit the transaction.

        Raises:
            Exception: An error occurred when adding the object to the database.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def delete(self):
        """
        Deletes the earthquake record from the database.
        """
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        The read method retrieves the object data from the object's attributes and returns it as a dictionary.

        Returns:
            dict: A dictionary containing the earthquake data.
        """
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth': self.depth,
            'time_of_day': self.time_of_day,
            'previous_magnitude': self.previous_magnitude,
            'distance_to_fault': self.distance_to_fault,
            'plate_boundary_type': self.plate_boundary_type,
            'soil_type': self.soil_type,
            'magnitude': self.magnitude
        }