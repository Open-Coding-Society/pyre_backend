import logging
from sqlite3 import IntegrityError
from __init__ import app, db

class Email(db.Model):
    __tablename__ = 'email'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    _date = db.Column(db.String(255), nullable=False)
    _email_id = db.Column(db.String(255), nullable=False)
    _subject = db.Column(db.String(255), nullable=False)
    _sender = db.Column(db.String(255), nullable=False)
    _description = db.Column(db.String(255), nullable=False)
    _hazard_type = db.Column(db.String(255), nullable=False)
    _location = db.Column(db.String(255), nullable=False)

    def __init__(self, date, email_id, subject, sender, description, hazard_type, location):
        self._date = date
        self._email_id = email_id
        self._subject = subject
        self._sender = sender
        self._description = description
        self._hazard_type = hazard_type
        self._location = location

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        return {
            'id': self.id,
            'date': self._date,
            'email_id': self._email_id,
            'subject': self._subject,
            'sender': self._sender,
            'description': self._description,
            'hazard_type': self._hazard_type,
            'location': self._location
        }

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    @staticmethod
    def restore(data):
        for email_data in data:
            _ = email_data.pop('id', None)  # Remove 'id' from post_data
            email_name = email_data.get("name", None)
            email = Email.query.filter_by(_name=email_name).first()
            if email:
                Email.update(email_data)
            else:
                email = Email(**email_data)
                email.update(email_data)
                email.create()

def initEmail():
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Tester data for table"""
        p1 = Email(date='04/25/2025', email_id='Unknown Email', subject='FIRE', sender="Unknown Email", description="Campfire left burning with no one around. Medium sized flames with dry brush nearby", hazard_type="Discarded Cigarette", location="Mt. Carmel High School")
        files = [p1]

        for file in files:
            try:
                file.create()
            except IntegrityError:
                '''fails with bad or duplicate data'''
                db.session.remove()