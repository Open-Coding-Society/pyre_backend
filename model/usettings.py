from __init__ import db, app

class Settings(db.Model):
    """
    Settings Model
    
    The Settings class represents the general settings of the application.
    
    Attributes:
        id (db.Column): The primary key, an integer representing the unique identifier for the record.
        description (db.Column): A string representing the description of the site.
        contact_email (db.Column): A string representing the contact email of the site.
        contact_phone (db.Column): A string representing the contact phone number of the site.
    """
    __tablename__ = 'settings'

    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(255), nullable=False)
    contact_email = db.Column(db.String(255), nullable=False)
    contact_phone = db.Column(db.String(20), nullable=False)

    def __init__(self, description, contact_email, contact_phone):
        self.description = description
        self.contact_email = contact_email
        self.contact_phone = contact_phone

    def __repr__(self):
        """
        The __repr__ method is a special method used to represent the object in a string format.
        Called by the repr() built-in function.
        
        Returns:
            str: A text representation of how to create the object.
        """
        return f"Settings(id={self.id}, description={self.description}, contact_email={self.contact_email}, contact_phone={self.contact_phone})"
    
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
        Deletes the settings from the database.
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
            dict: A dictionary containing the settings data.
        """
        return {
            'id': self.id,
            'description': self.description,
            'contact_email': self.contact_email,
            'contact_phone': self.contact_phone
        }

    @staticmethod
    def restore(data):
        """
        Restore settings from a list of dictionaries, replacing existing entries.

        Args:
            data (list): List of dictionaries containing settings data.
        
        Returns:
            dict: Dictionary of restored Settings objects.
        """
        with app.app_context():
            # Clear the existing table
            db.session.query(Settings).delete()
            db.session.commit()

            restored_settings = {}
            for settings_data in data:
                settings = Settings(
                    description=settings_data['description'],
                    contact_email=settings_data['contact_email'],
                    contact_phone=settings_data['contact_phone']
                )
                settings.create()
                restored_settings[settings_data['id']] = settings
            
            return restored_settings

def initSettings():
    """
    The initSettings function creates the Settings table and adds static data to the table.
    
    Uses:
        The db ORM methods to create the table.
    
    Instantiates:
        Settings objects with static data.
    
    Raises:
        Exception: An error occurred when adding the static data to the table.
    """
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Static data for table"""
        static_data = [
            Settings(description='A platform that evolves around your connections and creativity.', contact_email='pyre.ai.help@gmail.com', contact_phone='123-456-7890')
        ]
        
        for data in static_data:
            try:
                db.session.add(data)
                db.session.commit()
                print(f"Record created: {repr(data)}")
            except Exception as e:
                db.session.rollback()
                print(f"Error creating record for settings: {e}")

# Call initSettings to initialize the settings with static data
initSettings()