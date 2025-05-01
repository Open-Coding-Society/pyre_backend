import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from __init__ import app, db

class ForestFireModel:
    _instance = None

    @staticmethod
    def get_instance():
        if ForestFireModel._instance is None:
            ForestFireModel()
        return ForestFireModel._instance

    def __init__(self):
        if ForestFireModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ForestFireModel._instance = self
            # Initialize the model
            self.model = self._load_model()
            self.month_map = {'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,
                             'jul':6,'aug':7,'sep':8,'oct':9,'nov':10,'dec':11}
            self.day_map = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6}

    def _load_model(self):
        # In a real implementation, you would load a pre-trained model from disk
        # For this example, we'll train it on the fly
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
            df = pd.read_csv(url)

            df['month'] = df['month'].astype('category').cat.codes
            df['day'] = df['day'].astype('category').cat.codes

            X = df.drop('area', axis=1)
            y = df['area']

            # Apply log transformation to handle skewed distribution
            y_log = np.log1p(y)

            # Train a RandomForest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y_log)  # Train on all data for deployment

            return rf
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def estimate_FFMC(self, temp, rh, wind, rain):
        """Estimate Fine Fuel Moisture Code"""
        return max(18, min(101, 0.5 * temp - 0.3 * rh + 0.4 * wind - 0.5 * rain + 85))

    def estimate_DMC(self, temp, rh, rain):
        """Estimate Duff Moisture Code"""
        return max(0, 0.6 * temp - 0.35 * rh - 1.5 * rain + 33)

    def estimate_DC(self, temp, rain):
        """Estimate Drought Code"""
        return max(0, 0.5 * temp - 2 * rain + 100)

    def estimate_ISI(self, ffmc, wind):
        """Estimate Initial Spread Index"""
        return max(0, 0.208 * ffmc * wind / (90 + wind))

    def predict(self, forest_data):
        """
        Predict the burned area based on input forest and weather data

        Args:
            forest_data (dict): Dictionary containing input parameters

        Returns:
            dict: Dictionary with prediction results
        """
        try:
            # Process and prepare the input data
            temp = forest_data.get('temp', 0)
            rh = forest_data.get('RH', 0)
            wind = forest_data.get('wind', 0)
            rain = forest_data.get('rain', 0)
            month = forest_data.get('month', 'jan').lower()
            day = forest_data.get('day', 'mon').lower()
            x_coord = forest_data.get('X', 0)
            y_coord = forest_data.get('Y', 0)

            # Calculate fire indices if not provided
            ffmc = forest_data.get('FFMC', self.estimate_FFMC(temp, rh, wind, rain))
            dmc = forest_data.get('DMC', self.estimate_DMC(temp, rh, rain))
            dc = forest_data.get('DC', self.estimate_DC(temp, rain))
            isi = forest_data.get('ISI', self.estimate_ISI(ffmc, wind))

            # Convert month and day to numeric codes
            month_code = self.month_map.get(month, 0)
            day_code = self.day_map.get(day, 0)

            # Create input vector
            input_vector = pd.DataFrame([{
                'X': x_coord,
                'Y': y_coord,
                'month': month_code,
                'day': day_code,
                'FFMC': ffmc,
                'DMC': dmc,
                'DC': dc,
                'ISI': isi,
                'temp': temp,
                'RH': rh,
                'wind': wind,
                'rain': rain
            }])

            # Make prediction
            pred_log = self.model.predict(input_vector)[0]
            predicted_area = np.expm1(pred_log)  # Reverse log transformation

            return {
                "predicted_area": float(predicted_area),
                "indices": {
                    "FFMC": float(ffmc),
                    "DMC": float(dmc),
                    "DC": float(dc),
                    "ISI": float(isi)
                }
            }
        except Exception as e:
            return {"error": str(e)}


class ForestFire(db.Model):
    """
    ForestFire Model

    The ForestFire class represents a forest fire record with weather and location data.

    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the record.
        X (db.Column): An integer representing the X-axis spatial coordinate within the Montesinho park map.
        Y (db.Column): An integer representing the Y-axis spatial coordinate within the Montesinho park map.
        month (db.Column): A string representing the month of the year.
        day (db.Column): A string representing the day of the week.
        FFMC (db.Column): A float representing the Fine Fuel Moisture Code index.
        DMC (db.Column): A float representing the Duff Moisture Code index.
        DC (db.Column): A float representing the Drought Code index.
        ISI (db.Column): A float representing the Initial Spread Index.
        temp (db.Column): A float representing the temperature in Celsius degrees.
        RH (db.Column): An integer representing the relative humidity in percentage.
        wind (db.Column): A float representing the wind speed in km/h.
        rain (db.Column): A float representing the outside rain in mm/m².
        area (db.Column): A float representing the burned area of the forest in hectares.
    """
    __tablename__ = 'forest_fires'

    id = db.Column(db.Integer, primary_key=True)
    X = db.Column(db.Integer, nullable=False)
    Y = db.Column(db.Integer, nullable=False)
    month = db.Column(db.String(3), nullable=False)
    day = db.Column(db.String(3), nullable=False)
    FFMC = db.Column(db.Float, nullable=False)
    DMC = db.Column(db.Float, nullable=False)
    DC = db.Column(db.Float, nullable=False)
    ISI = db.Column(db.Float, nullable=False)
    temp = db.Column(db.Float, nullable=False)
    RH = db.Column(db.Integer, nullable=False)
    wind = db.Column(db.Float, nullable=False)
    rain = db.Column(db.Float, nullable=False)
    area = db.Column(db.Float, nullable=False)

    def __init__(self, X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area=0.0):
        """
        Constructor, initializes a ForestFire object.

        Args:
            X (int): The X-axis spatial coordinate within the Montesinho park map.
            Y (int): The Y-axis spatial coordinate within the Montesinho park map.
            month (str): The month of the year ('jan' to 'dec').
            day (str): The day of the week ('mon' to 'sun').
            FFMC (float): The Fine Fuel Moisture Code index.
            DMC (float): The Duff Moisture Code index.
            DC (float): The Drought Code index.
            ISI (float): The Initial Spread Index.
            temp (float): The temperature in Celsius degrees.
            RH (int): The relative humidity in percentage.
            wind (float): The wind speed in km/h.
            rain (float): The outside rain in mm/m².
            area (float, optional): The burned area of the forest in hectares. Defaults to 0.0.
        """
        self.X = X
        self.Y = Y
        self.month = month
        self.day = day
        self.FFMC = FFMC
        self.DMC = DMC
        self.DC = DC
        self.ISI = ISI
        self.temp = temp
        self.RH = RH
        self.wind = wind
        self.rain = rain
        self.area = area

    def __repr__(self):
        """
        The __repr__ method is a special method used to represent the object in a string format.
        Called by the repr() built-in function.

        Returns:
            str: A text representation of how to create the object.
        """
        return f"ForestFire(id={self.id}, X={self.X}, Y={self.Y}, month='{self.month}', day='{self.day}', FFMC={self.FFMC}, DMC={self.DMC}, DC={self.DC}, ISI={self.ISI}, temp={self.temp}, RH={self.RH}, wind={self.wind}, rain={self.rain}, area={self.area})"

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
        Deletes the forest fire record from the database.
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
            dict: A dictionary containing the forest fire data.
        """
        return {
            'id': self.id,
            'X': self.X,
            'Y': self.Y,
            'month': self.month,
            'day': self.day,
            'FFMC': self.FFMC,
            'DMC': self.DMC,
            'DC': self.DC,
            'ISI': self.ISI,
            'temp': self.temp,
            'RH': self.RH,
            'wind': self.wind,
            'rain': self.rain,
            'area': self.area
        }