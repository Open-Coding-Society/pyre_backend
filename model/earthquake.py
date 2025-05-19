from __init__ import app, db

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
