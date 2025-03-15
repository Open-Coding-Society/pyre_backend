from __init__ import db, app
from sqlalchemy.exc import IntegrityError
from model.user import User

class TopInterest(db.Model):
    """
    TopInterest Model

    Represents an interest and the number of users who share it.
    """
    __tablename__ = 'top_interests'

    _interests = db.Column(db.String(50), primary_key=True, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __init__(self, _interests, count):
        """
        Initialize a TopInterest instance.
        """
        self._interests = _interests
        self.count = count

    def create(self):
        """
        Add the TopInterest instance to the database and commit.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        Retrieve the TopInterest instance data as a dictionary.
        """
        return {
            "_interests": self._interests,
            "count": self.count
        }

    def update(self, data):
        """
        Update the TopInterest instance with the provided data dictionary.
        """
        try:
            self._interests = data.get('_interests', self._interests)
            self.count = data.get('count', self.count)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def delete(self):
        """
        Remove the TopInterest instance from the database and commit.
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
        Restore TopInterest instances from a list of dictionaries.
        """
        restored_interests = {}
        for interest_data in data:
            try:
                _interests = interest_data.get("_interests", None)
                count = interest_data.get("count", None)

                if not _interests or count is None:
                    continue

                interest_key = _interests
                top_interest = TopInterest.query.filter_by(_interests=_interests).first()
                if top_interest:
                    top_interest.update(interest_data)
                else:
                    top_interest = TopInterest(**interest_data)
                    top_interest.create()

                restored_interests[interest_key] = top_interest
            except Exception as e:
                print(f"Error processing interest data: {interest_data} - {e}")
                continue

        return restored_interests


def initTopInterests():
    """
    Initialize the TopInterest table with default data.
    """
    with app.app_context():
        db.create_all()

        top_interests = [
            {"_interests": "Programming", "count": 10},
            {"_interests": "Reading", "count": 8},
            {"_interests": "Physics", "count": 6},
            {"_interests": "Mathematics", "count": 5},
            {"_interests": "Nature", "count": 4},
        ]

        for interest_data in top_interests:
            try:
                top_interest = TopInterest(**interest_data)
                db.session.add(top_interest)
                db.session.commit()
                print(f"Added interest: {top_interest._interests}")
            except IntegrityError:
                db.session.rollback()
                print(f"Duplicate or error for interest: {interest_data['_interests']}")


def updateTopInterests():
    """
    Dynamically update the TopInterest table based on user data.
    """
    with app.app_context():
        all_users = User.query.all()
        interest_counts = {}

        for user in all_users:
            user_interests = user._interests.split(", ")
            for interest in user_interests:
                interest_counts[interest] = interest_counts.get(interest, 0) + 1

        for _interests, count in interest_counts.items():
            top_interest = TopInterest.query.filter_by(_interests=_interests).first()
            if top_interest:
                top_interest.update({"_interests": _interests, "count": count})
            else:
                try:
                    top_interest = TopInterest(_interests=_interests, count=count)
                    db.session.add(top_interest)
                    db.session.commit()
                    print(f"Added interest: {top_interest._interests} with count: {top_interest.count}")
                except IntegrityError:
                    db.session.rollback()
                    print(f"Error adding interest: {_interests}")
