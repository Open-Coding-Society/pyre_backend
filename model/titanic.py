import numpy as np
from __init__ import app, db

class TitanicModel:
    _instance = None

    @staticmethod
    def get_instance():
        if TitanicModel._instance is None:
            TitanicModel()
        return TitanicModel._instance

    def __init__(self):
        if TitanicModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            TitanicModel._instance = self
            # Initialize your model here
            # For example, load a pre-trained model or set up parameters
            self.model = self._load_model()

    def _load_model(self):
        # Load or initialize your model here
        # This is just a placeholder for the actual model loading code
        return None

    def predict(self, passenger):
        # Implement your prediction logic here
        # This is just a placeholder for the actual prediction code
        # For example, you can use the model to predict the survival probability
        # based on the passenger data
        survival_probability = np.random.rand()  # Random probability for demonstration
        return {"survival_probability": survival_probability}

class Passenger(db.Model):
    """
    Passenger Model
    
    The Passenger class represents a Titanic passenger.
    
    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the record.
        name (db.Column): A string representing the name of the passenger.
        age (db.Column): An integer representing the age of the passenger.
        sex (db.Column): A string representing the sex of the passenger.
        survived (db.Column): A boolean representing if the passenger survived.
        pclass (db.Column): An integer representing the passenger's class (1, 2, or 3).
        sibsp (db.Column): An integer representing the number of siblings/spouses aboard.
        parch (db.Column): An integer representing the number of parents/children aboard.
        fare (db.Column): A float representing the fare the passenger paid.
        embarked (db.Column): A string representing the port at which the passenger embarked ('C', 'Q', or 'S').
        alone (db.Column): A boolean representing whether the passenger is alone.
    """
    __tablename__ = 'passengers'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(50), nullable=False)
    survived = db.Column(db.Boolean, nullable=False)
    pclass = db.Column(db.Integer, nullable=False)
    sibsp = db.Column(db.Integer, nullable=False)
    parch = db.Column(db.Integer, nullable=False)
    fare = db.Column(db.Float, nullable=False)
    embarked = db.Column(db.String(1), nullable=False)
    alone = db.Column(db.Boolean, nullable=False)

    def __init__(self, name, age, sex, survived, pclass, sibsp, parch, fare, embarked, alone):
        """
        Constructor, initializes a Passenger object.
        
        Args:
            name (str): The name of the passenger.
            age (int): The age of the passenger.
            sex (str): The sex of the passenger.
            survived (bool): Whether the passenger survived.
            pclass (int): The passenger's class (1, 2, or 3).
            sibsp (int): The number of siblings/spouses aboard.
            parch (int): The number of parents/children aboard.
            fare (float): The fare the passenger paid.
            embarked (str): The port at which the passenger embarked ('C', 'Q', or 'S').
            alone (bool): Whether the passenger is alone.
        """
        self.name = name
        self.age = age
        self.sex = sex
        self.survived = survived
        self.pclass = pclass
        self.sibsp = sibsp
        self.parch = parch
        self.fare = fare
        self.embarked = embarked
        self.alone = alone

    def __repr__(self):
        """
        The __repr__ method is a special method used to represent the object in a string format.
        Called by the repr() built-in function.
        
        Returns:
            str: A text representation of how to create the object.
        """
        return f"Passenger(id={self.id}, name={self.name}, age={self.age}, sex={self.sex}, survived={self.survived}, pclass={self.pclass}, sibsp={self.sibsp}, parch={self.parch}, fare={self.fare}, embarked={self.embarked}, alone={self.alone})"
    
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
        Deletes the passenger from the database.
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
            dict: A dictionary containing the passenger data.
        """
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'sex': self.sex,
            'survived': self.survived,
            'pclass': self.pclass,
            'sibsp': self.sibsp,
            'parch': self.parch,
            'fare': self.fare,
            'embarked': self.embarked,
            'alone': self.alone
        }

def initPassengers():
    """
    The initPassengers function creates the Passengers table and adds tester data to the table.
    
    Uses:
        The db ORM methods to create the table.
    
    Instantiates:
        Passenger objects with tester data.
    
    Raises:
        Exception: An error occurred when adding the tester data to the table.
    """
    with app.app_context():
        db.create_all()
        """Tester data for table"""
        tester_data = [
            Passenger(name='John Doe', age=30, sex='male', survived=True, pclass=1, sibsp=0, parch=0, fare=100.0, embarked='C', alone=True),
            Passenger(name='Jane Doe', age=25, sex='female', survived=False, pclass=2, sibsp=1, parch=0, fare=50.0, embarked='Q', alone=False),
            Passenger(name='Alice', age=22, sex='female', survived=True, pclass=3, sibsp=0, parch=1, fare=30.0, embarked='S', alone=False)
        ]
        
        for data in tester_data:
            try:
                db.session.add(data)
                db.session.commit()
                print(f"Record created: {repr(data)}")
            except Exception as e:
                db.session.rollback()
                print(f"Error creating record for passenger {data.name}: {e}")