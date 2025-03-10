from __init__ import db, app
from sqlalchemy.exc import IntegrityError
from model.post import Post
from model.user import User

class Vote(db.Model):
    """
    Vote Model

    The Vote class represents a single upvote or downvote on a post by a user.

    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the vote.
        _vote_type (db.Column): A string representing the type of vote ("upvote" or "downvote").
        _user_id (db.Column): An integer representing the ID of the user who cast the vote.
        _post_id (db.Column): An integer representing the ID of the post that received the vote.
    """
    __tablename__ = 'votes'

    id = db.Column(db.Integer, primary_key=True)
    _vote_type = db.Column(db.String(10), nullable=False)  # "upvote" or "downvote"
    _user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    _post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=False)

    def __init__(self, vote_type, user_id, post_id):
        """
        Constructor to initialize a vote.

        Args:
            vote_type (str): Type of the vote, either "upvote" or "downvote".
            user_id (int): ID of the user who cast the vote.
            post_id (int): ID of the post that received the vote.
        """
        self._vote_type = vote_type
        self._user_id = user_id
        self._post_id = post_id

    def create(self):
        """
        Add the vote to the database and commit the transaction.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        Retrieve the vote data as a dictionary.

        Returns:
            dict: Dictionary with vote information.
        """
        return {
            "id": self.id,
            "vote_type": self._vote_type,
            "user_id": self._user_id,
            "post_id": self._post_id
        }
    
    def update(self, vote_type):
        """
        Update the vote type and commit the transaction.

        Args:
            vote_type (str): Type of the vote, either "upvote" or "downvote".
        """
        try:
            self._vote_type = vote_type
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise

    def delete(self):
        """
        Remove the vote from the database and commit the transaction.
        """
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        
    @staticmethod
    def restore(data):
        """
        Restore votes from a list of dictionaries, ensuring no duplicates are created.

        Args:
            data (list): List of dictionaries containing vote data.

        Returns:
            list: List of restored Vote objects.
        """
        restored_classes = {}

        for vote_data in data:
            existing_vote = Vote.query.filter_by(
                _user_id=vote_data['user_id'], _post_id=vote_data['post_id']
            ).first()

            if existing_vote:
                # If vote exists, update it if needed (optional)
                if existing_vote._vote_type != vote_data['vote_type']:
                    existing_vote.update(vote_data['vote_type'])
                restored_classes[vote_data['id']] = existing_vote
            else:
                # If vote doesn't exist, create a new one
                vote = Vote(vote_data['vote_type'], vote_data['user_id'], vote_data['post_id'])
                vote.create()
                restored_classes[vote_data['id']] = vote

        return restored_classes



def initVotes():
    """
    Initialize the Vote table with any required starter data.
    """
    with app.app_context():
        # Create database tables if they don't exist
        db.create_all()

        # Optionally, add some test data (replace with actual values as needed)
        votes = [
            Vote(vote_type='upvote', user_id=1, post_id=5),
            Vote(vote_type='downvote', user_id=2, post_id=5),
            Vote(vote_type='downvote', user_id=3, post_id=7),
            Vote(vote_type='downvote', user_id=7, post_id=6),
        ]
        
        for vote in votes:
            try:
                db.session.add(vote)
                db.session.commit()
                print(f"Record created: {repr(vote)}")
            except IntegrityError:
                db.session.rollback()
                print(f"Duplicate or error: {repr(vote)}")
