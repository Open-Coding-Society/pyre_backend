from __init__ import db, app
from sqlalchemy.exc import IntegrityError
from model.user import User

class TopUser(db.Model):
    """
    TopUser Model

    The TopUser class represents a user with their interests and a score.
    """
    __tablename__ = 'top_users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    interests = db.Column(db.String(255), nullable=False)
    score = db.Column(db.Integer, nullable=False)

    def __init__(self, name, interests, score):
        # Initialize the TopUser instance with name, interests, and score
        self.name = name
        self.interests = interests
        self.score = score

    def create(self):
        try:
            # Add the top user to the database and commit the transaction
            db.session.add(self)
            db.session.commit()
        except IntegrityError:
            # Rollback the transaction if there is an integrity error (e.g., duplicate entry of user)
            db.session.rollback()

    def read(self):
        # Return the top user data as a dictionary
        return {
            "id": self.id,
            "name": self.name,
            "interests": self.interests,
            "score": self.score
        }

    def delete(self):
        try:
            # Delete the top user from the database and commit the transaction
            db.session.delete(self)
            db.session.commit()
        except IntegrityError:
            # Rollback the transaction if there is an integrity error
            db.session.rollback()

def initTopUsers():
    with app.app_context():
        # Create database tables if they don't exist
        db.create_all()
        # Initialize the TopUser table with some static data
        top_users = [
            TopUser(name='Alice', interests='reading, hiking', score=10),
            TopUser(name='Bob', interests='cooking, hiking', score=8),
        ]
        # Add each top user to the database
        for top_user in top_users:
            top_user.create()