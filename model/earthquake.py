from __init__ import app, db
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from __init__ import app, db
from sqlalchemy.exc import IntegrityError

class Earthquake(db.Model):
    """Earthquake database model"""
    __tablename__ = 'earthquakes'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Core earthquake data
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    time_of_day = db.Column(db.String(50), nullable=False)
    previous_magnitude = db.Column(db.Float, nullable=False)
    distance_to_fault = db.Column(db.Float, nullable=False)
    plate_boundary_type = db.Column(db.String(50), nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    magnitude = db.Column(db.Float, nullable=False)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(255))
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    def __init__(self, latitude, longitude, depth, time_of_day, previous_magnitude,
                 distance_to_fault, plate_boundary_type, soil_type, magnitude):
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time_of_day = time_of_day
        self.previous_magnitude = previous_magnitude
        self.distance_to_fault = distance_to_fault
        self.plate_boundary_type = plate_boundary_type
        self.soil_type = soil_type
        self.magnitude = magnitude
        self.created_by = "NathanTejidor"  # Current user's login

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            raise Exception("Error occurred while creating earthquake record")

    def read(self):
        return {
            "id": self.id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth": self.depth,
            "time_of_day": self.time_of_day,
            "previous_magnitude": self.previous_magnitude,
            "distance_to_fault": self.distance_to_fault,
            "plate_boundary_type": self.plate_boundary_type,
            "soil_type": self.soil_type,
            "magnitude": self.magnitude,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def update(self, data):
        try:
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error occurred while updating earthquake record: {str(e)}")

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error occurred while deleting earthquake record: {str(e)}")

class EarthquakeModel:
    """Singleton class for earthquake prediction model"""
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
            # Initialize the model and mappings
            self.model = self._load_model()
            self.time_map = {
                'morning': 0,
                'afternoon': 1,
                'evening': 2,
                'night': 3
            }
            self.plate_type_map = {
                'convergent': 0,
                'divergent': 1,
                'transform': 2,
                'intraplate': 3
            }
            self.soil_type_map = {
                'rock': 0,
                'stiff_soil': 1,
                'soft_soil': 2,
                'alluvium': 3
            }

    def _load_model(self):
        """Load the pre-trained model from file"""
        try:
            model = joblib.load('earthquake_model.pkl')
            print(f"Model loaded successfully at {datetime.utcnow().isoformat()}")
            return model
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}")

    def predict(self, earthquake_data):
        """Make predictions using the loaded model"""
        try:
            # Validate input data
            self._validate_input_data(earthquake_data)
            
            # Prepare features for prediction
            features = self._prepare_features(earthquake_data)
            
            # Make prediction
            predicted_magnitude = self.model.predict(features)[0]
            
            # Calculate confidence score (if available in your model)
            confidence_score = self._calculate_confidence(features)
            
            return {
                "predicted_magnitude": float(predicted_magnitude),
                "confidence_score": confidence_score,
                "timestamp": datetime.utcnow().isoformat(),
                "input_parameters": earthquake_data
            }
        except KeyError as e:
            return {"error": f"Missing or invalid field: {e}"}
        except Exception as e:
            return {"error": f"An error occurred during prediction: {e}"}

    def _validate_input_data(self, data):
        """Validate input data for prediction"""
        required_fields = [
            'latitude', 'longitude', 'depth', 'time_of_day',
            'previous_magnitude', 'distance_to_fault',
            'plate_boundary_type', 'soil_type'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
            
        # Validate value ranges
        if not (-90 <= data['latitude'] <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= data['longitude'] <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        if data['depth'] < 0:
            raise ValueError("Depth must be non-negative")
        if data['time_of_day'] not in self.time_map:
            raise ValueError(f"Invalid time_of_day. Must be one of: {', '.join(self.time_map.keys())}")
        if data['plate_boundary_type'] not in self.plate_type_map:
            raise ValueError(f"Invalid plate_boundary_type. Must be one of: {', '.join(self.plate_type_map.keys())}")
        if data['soil_type'] not in self.soil_type_map:
            raise ValueError(f"Invalid soil_type. Must be one of: {', '.join(self.soil_type_map.keys())}")

    def _prepare_features(self, earthquake_data):
        """Prepare features for model prediction"""
        return np.array([
            earthquake_data['latitude'],
            earthquake_data['longitude'],
            earthquake_data['depth'],
            self.time_map[earthquake_data['time_of_day']],
            earthquake_data['previous_magnitude'],
            earthquake_data['distance_to_fault'],
            self.plate_type_map[earthquake_data['plate_boundary_type']],
            self.soil_type_map[earthquake_data['soil_type']]
        ]).reshape(1, -1)

    def _calculate_confidence(self, features):
        """Calculate confidence score for prediction"""
        try:
            # If your model supports probabilistic predictions, use those
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
                return float(np.max(probabilities))
            # For random forests, we can use standard deviation of trees
            elif isinstance(self.model, RandomForestRegressor):
                predictions = [tree.predict(features)[0] for tree in self.model.estimators_]
                return 1 - float(np.std(predictions))
            else:
                # Default confidence if no better method is available
                return 0.85
        except Exception:
            return 0.85

    def calculate_risk_index(self, magnitude, depth, distance_to_fault, soil_type):
        """Calculate earthquake risk index"""
        try:
            # Normalize factors
            magnitude_factor = magnitude / 10.0  # Assuming max magnitude of 10
            depth_factor = 1 - (min(depth, 700) / 700)  # Assuming max depth of 700km
            distance_factor = 1 - (min(distance_to_fault, 1000) / 1000)  # Assuming max distance of 1000km
            soil_factor = self._get_soil_risk_factor(soil_type)

            # Calculate weighted risk index
            risk_index = (
                magnitude_factor * 0.4 +
                depth_factor * 0.25 +
                distance_factor * 0.2 +
                soil_factor * 0.15
            )

            return min(max(risk_index, 0), 1)  # Ensure result is between 0 and 1
        except Exception as e:
            raise Exception(f"Error calculating risk index: {str(e)}")

    def _get_soil_risk_factor(self, soil_type):
        """Get risk factor based on soil type"""
        soil_risk_factors = {
            'rock': 0.2,
            'stiff_soil': 0.4,
            'soft_soil': 0.7,
            'alluvium': 0.9
        }
        return soil_risk_factors.get(soil_type.lower(), 0.5)

    def get_risk_level(self, risk_index):
        """Convert risk index to descriptive risk level"""
        if risk_index < 0.2:
            return "Very Low"
        elif risk_index < 0.4:
            return "Low"
        elif risk_index < 0.6:
            return "Moderate"
        elif risk_index < 0.8:
            return "High"
        else:
            return "Very High"