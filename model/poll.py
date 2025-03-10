from __init__ import db

class Poll(db.Model):
    """
    Poll Model

    This model represents a poll with a name and optional interests.
    """
    __tablename__ = 'polls'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    interests = db.Column(db.String(500), nullable=False)

    def __init__(self, name, interests):
        self.name = name
        self.interests = interests



    def create(self):
        """
        Add the poll to the database and commit.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        Retrieve the poll's data as a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "interests": self.interests
        }

    def update(self, data):
        """
        Update the school class's data with the provided dictionary.
        """
        try:
            self.name = data.get('name', self.name)
            self.interests = data.get('interests', self.interests)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e


    def delete(self):
        """
        Remove the poll from the database and commit.
        """
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        
    @staticmethod
    def restore(data):
        # Restore polls using a list of dictionaries.
        restored_polls = {}
        for poll_data in data:
            try:
                _ = poll_data.pop('id', None) # remove id column from poll_data
                name = poll_data.get("name", None)
                interests = poll_data.get("interests", None)

                poll_key = name
                poll = Poll.query.filter_by(name=name).first()
                if poll:
                    poll.update(poll_data)
                else:
                    poll = Poll(**poll_data)
                    poll.create()

                restored_polls[poll_key] = poll
            except Exception as e:
                print(f"Error processing poll data: {poll_data} - {e}")
                continue

        return restored_polls


def initPolls():
    """
    Initialize the Poll table with default data.
    """
    polls = [
        Poll("Thomas Edison", "Favorite genre of music: Jazz"),
        Poll("Grace Hopper", "Favorite genre of music: Rock")
    ]
    for poll in polls:
        try:
            db.session.add(poll)
            db.session.commit()
            print(f"Added poll: {poll.name}")
        except Exception as e:
            db.session.rollback()
            print(f"Error adding poll: {poll.name} - {e}")

