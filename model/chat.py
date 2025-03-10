import logging
from sqlite3 import IntegrityError
from sqlalchemy import JSON
from sqlalchemy.exc import IntegrityError
from __init__ import app, db
from model.user import User
from model.channel import Channel

class Chat(db.Model):
    """
    Chat Model

    The Chat class represents individual messages in a chat channel.

    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the chat message.
        _message (db.Column): A string representing the content of the chat message.
        _user_id (db.Column): An integer representing the user who sent the message.
        _channel_id (db.Column): An integer representing the channel to which the message belongs.
    """
    __tablename__ = 'chats'

    id = db.Column(db.Integer, primary_key=True)
    _message = db.Column(db.String(255), nullable=False)
    _user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    _channel_id = db.Column(db.Integer, db.ForeignKey('channels.id'), nullable=False)

    def __init__(self, message, user_id=None, channel_id=None):
        """
        Constructor for Chat Model.

        Args:
            message (str): The content of the chat message.
            user_id (int): The ID of the user sending the message.
            channel_id (int): The ID of the channel where the message is sent.
        """
        self._message = message
        self._user_id = user_id
        self._channel_id = channel_id

    def __repr__(self):
        """
        Represents the Chat object in a string format.

        Returns:
            str: A text representation of how to create the object.
        """
        return f"Chat(id={self.id}, message={self._message}, user_id={self._user_id}, channel_id={self._channel_id})"

    def create(self):
        """
        Creates a new chat message in the database.

        Returns:
            Chat: The created chat object, or None on error.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except IntegrityError as e:
            db.session.rollback()
            logging.warning(f"IntegrityError: Could not create chat with message '{self._message}' due to {str(e)}.")
            return None
        return self

    def read(self):
        """
        Retrieves chat message data as a dictionary.

        Returns:
            dict: A dictionary containing the chat message data, including user and channel names.
        """
        user = User.query.get(self._user_id)
        channel = Channel.query.get(self._channel_id)
        return {
            "id": self.id,
            "message": self._message,
            "user_id": user.id if user else None,
            "channel_id": channel.id if channel else None
        }

    def update(self, data):
        """
        Updates the chat message object with new data.

        Args:
            data (dict): A dictionary containing the new data for the chat message.

        Returns:
            Chat: The updated chat message object, or None on error.
        """
        if 'message' in data:
            self._message = data['message']

        try:
            db.session.commit()
        except IntegrityError as e:
            db.session.rollback()
            logging.warning(f"IntegrityError: Could not update chat with ID '{self.id}' due to {str(e)}.")
            return None
        return self

    def delete(self):
        """
        Deletes the chat message from the database.
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
        Restore chats from a list of dictionaries.
        Args:
            data (list): A list of dictionaries containing chat data.
        Returns:
            dict: A dictionary of restored chats keyed by message ID.
        """
        restored_chats = {}
        for chat_data in data:
            _ = chat_data.pop('id', None)  # Remove 'id' from chat_data if present
            message = chat_data.get("message", None)
            chat = Chat.query.filter_by(_message=message).first()
            if chat:
                chat.update(chat_data)
            else:
                chat = Chat(**chat_data)
                chat.create()
            restored_chats[message] = chat
        return restored_chats

def initChats():
    """
    Initializes the Chat table and adds tester data to the table.

    Uses:
        The db ORM methods to create the table.

    Instantiates:
        Chat objects with tester data.

    Raises:
        IntegrityError: An error occurred when adding the tester data to the table.
    """
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Tester data for table"""
        chats = [
            Chat(message="Hi!", user_id=1, channel_id=8),
            Chat(message="Hello, how are you Toby?", user_id=2, channel_id=8),
            Chat(message="I'm good. What are your opinions on the new physics law that was just discovered?", user_id=1, channel_id=8),
            Chat(message="I think its quite amazing. Let's discuss the project.", user_id=2, channel_id=8),
            Chat(message="How can this impact the world later on?", user_id=1, channel_id=8),
            Chat(message="Not sure... but whatever it does, it will be MASSIVE. I know it.", user_id=2, channel_id=8),
        ]

        for chat in chats:
            try:
                chat.create()
                print(f"Record created: {repr(chat)}")
            except IntegrityError:
                db.session.remove()
                print(f"Record exists or error: {chat._message}")
    print("All non test chats initiated")
