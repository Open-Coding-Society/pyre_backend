import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from __init__ import app, db
from datetime import datetime

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

    def _load_model(self):
        # Load and process the earthquakes.csv data
        try:
            df = pd.read_csv('earthquakes.csv')
            
            # Convert time to datetime features
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            df['day'] = df['time'].dt.day
            df['month'] = df['time'].dt.month
            
            # Select features for prediction
            features = ['latitude', 'longitude', 'depth', 'hour', 'day', 'month']
            X = df[features]
            y = df['mag']  # Predict magnitude

            # Train a RandomForest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)  # Train on all data for deployment

            return rf
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, earthquake_data):
        """
        Predict the magnitude based on location and time data

        Args:
            earthquake_data (dict): Dictionary containing input parameters

        Returns:
            dict: Dictionary with prediction results
        """
        try:
            # Process and prepare the input data
            time = pd.to_datetime(earthquake_data.get('time', pd.Timestamp.now()))
            lat = earthquake_data.get('latitude', 0)
            lon = earthquake_data.get('longitude', 0)
            depth = earthquake_data.get('depth', 0)

            # Create input vector
            input_vector = pd.DataFrame([{
                'latitude': lat,
                'longitude': lon,
                'depth': depth,
                'hour': time.hour,
                'day': time.day,
                'month': time.month
            }])

            # Make prediction
            predicted_magnitude = self.model.predict(input_vector)[0]

            return {
                "predicted_magnitude": float(predicted_magnitude),
                "location": {
                    "latitude": lat,
                    "longitude": lon,
                    "depth": depth
                },
                "time_features": {
                    "hour": time.hour,
                    "day": time.day,
                    "month": time.month
                }
            }
        except Exception as e:
            return {"error": str(e)}


class Earthquake(db.Model):
    """
    Earthquake Model

    The Earthquake class represents an earthquake record with location, time, and measurement data.

    Attributes:
        id (db.Column): The primary key
        time (db.Column): The time of the earthquake
        latitude (db.Column): Latitude of the epicenter
        longitude (db.Column): Longitude of the epicenter
        depth (db.Column): Depth in kilometers
        mag (db.Column): Magnitude of the earthquake
        magType (db.Column): Type of magnitude measurement
        place (db.Column): Description of the location
        type (db.Column): Type of seismic event
        soil_type (db.Column): Type of soil at the location
        plate_boundary_type (db.Column): Type of plate boundary
        previous_magnitude (db.Column): Previous magnitude in the area
        distance_to_fault (db.Column): Distance to nearest fault line
    """
    __tablename__ = 'earthquakes'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    mag = db.Column(db.Float, nullable=False)
    magType = db.Column(db.String(10))
    place = db.Column(db.String(200))
    type = db.Column(db.String(50))
    soil_type = db.Column(db.String(50), default='Unknown')
    plate_boundary_type = db.Column(db.String(50), default='Transform')
    previous_magnitude = db.Column(db.Float, default=0.0)
    distance_to_fault = db.Column(db.Float, default=0.0)

    def __init__(self, time, latitude, longitude, depth, mag, magType=None, place=None, type="earthquake",
                 soil_type="Unknown", plate_boundary_type="Transform", previous_magnitude=0.0, distance_to_fault=0.0):
        """
        Constructor for Earthquake

        Args:
            time (datetime): Time of the earthquake
            latitude (float): Latitude of the epicenter
            longitude (float): Longitude of the epicenter
            depth (float): Depth in kilometers
            mag (float): Magnitude of the earthquake
            magType (str, optional): Type of magnitude measurement
            place (str, optional): Description of the location
            type (str, optional): Type of seismic event, defaults to "earthquake"
            soil_type (str, optional): Type of soil at the location
            plate_boundary_type (str, optional): Type of plate boundary
            previous_magnitude (float, optional): Previous magnitude in the area
            distance_to_fault (float, optional): Distance to nearest fault line
        """
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.mag = mag
        self.magType = magType
        self.place = place
        self.type = type
        self.soil_type = soil_type
        self.plate_boundary_type = plate_boundary_type
        self.previous_magnitude = previous_magnitude
        self.distance_to_fault = distance_to_fault

    def __repr__(self):
        """String representation of the Earthquake object"""
        return f"Earthquake(time={self.time}, latitude={self.latitude}, longitude={self.longitude}, depth={self.depth}, mag={self.mag}, place='{self.place}')"

    def create(self):
        """Add this earthquake record to the database"""
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def delete(self):
        """Delete this earthquake record from the database"""
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """Return a dictionary representation of the earthquake record"""
        return {
            'id': self.id,
            'time': self.time.isoformat() if self.time else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth': self.depth,
            'magnitude': self.mag,  # Changed from mag to magnitude
            'magType': self.magType,
            'place': self.place,
            'type': self.type,
            'time_of_day': self.time.hour if self.time else 0,
            'soil_type': self.soil_type,
            'plate_boundary_type': self.plate_boundary_type,
            'previous_magnitude': self.previous_magnitude,
            'distance_to_fault': self.distance_to_fault
        }

def initEarthquakes():
    """Initialize the earthquakes table with data from earthquakes.csv"""
    try:
        # Check if we already have data
        if Earthquake.query.first():
            return
        
        # Read the CSV file
        df = pd.read_csv('earthquakes.csv')
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Create Earthquake records
        for _, row in df.iterrows():
            earthquake = Earthquake(
                time=row['time'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                depth=row['depth'],
                mag=row['mag'],
                magType=row.get('magType', None),
                place=row.get('place', None),
                type=row.get('type', 'earthquake'),
                soil_type=row.get('soil_type', 'Unknown'),
                plate_boundary_type=row.get('plate_boundary_type', 'Transform'),
                previous_magnitude=row.get('previous_magnitude', 0.0),
                distance_to_fault=row.get('distance_to_fault', 0.0)
            )
            db.session.add(earthquake)
        
        # Commit all records
        db.session.commit()
        print("Earthquake data initialized successfully")
    except Exception as e:
        db.session.rollback()
        print(f"Error initializing earthquake data: {e}")