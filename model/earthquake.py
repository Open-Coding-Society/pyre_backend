from __init__ import app, db
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class Earthquake(db.Model):
    """
    Earthquake Model

    Represents a significant earthquake event with relevant attributes.
    """
    __tablename__ = 'earthquakes'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    mag = db.Column(db.Float, nullable=False)
    magType = db.Column(db.String, nullable=True)
    place = db.Column(db.String, nullable=True)
    type = db.Column(db.String, nullable=False)

    def __init__(self, time, latitude, longitude, depth, mag, magType, place, type):
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.mag = mag
        self.magType = magType
        self.place = place
        self.type = type

    def __repr__(self):
        return f"Earthquake(id={self.id}, time='{self.time}', lat={self.latitude}, lon={self.longitude}, mag={self.mag})"

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        return {
            'id': self.id,
            'time': self.time,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth': self.depth,
            'mag': self.mag,
            'magType': self.magType,
            'place': self.place,
            'type': self.type
        }


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
            self.model = self._load_model()

    def _load_model(self):
        """Load and train a RandomForest model on earthquake data"""
        try:
            # Load the local earthquake CSV file
            file_path = "earthquakes.csv"  # Ensure this path exists in your project root
            df = pd.read_csv(file_path)

            # Filter out rows with missing target
            df = df[df['mag'].notnull()]

            # Convert categorical column
            df['magType'] = df['magType'].astype('category').cat.codes

            # Select features and target
            features = [
                'latitude', 'longitude', 'depth', 'magType', 'nst',
                'gap', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst'
            ]
            df = df[features + ['mag']].dropna()

            X = df[features]
            y = df['mag']

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            return model
        except Exception as e:
            print(f"Error loading earthquake model: {e}")
            return None
        