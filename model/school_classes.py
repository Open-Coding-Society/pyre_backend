from __init__ import db

class SchoolClass(db.Model):
    """
    SchoolClass Model

    This model represents a school class with its subject and teacher information.
    """
    __tablename__ = 'school_classes'

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(100), nullable=False)
    teacher = db.Column(db.String(500), nullable=True)  # Comma-separated list of teachers
    building = db.Column(db.String(100), nullable=True)  # Building where the class is located

    def __init__(self, subject, teacher, building=None):
        self.subject = subject
        self.teacher = ', '.join(teacher) if isinstance(teacher, list) else teacher
        self.building = building

    def create(self):
        """
        Add the school class to the database and commit the transaction.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        Retrieve the school class's data as a dictionary.
        """
        return {
            "id": self.id,
            "subject": self.subject,
            "teacher": self.teacher.split(', ') if self.teacher else [],
            "building": self.building
        }

    def update(self, data):
        """
        Update the school class's data with the provided dictionary.
        """
        try:
            self.subject = data.get('subject', self.subject)
            teacher = data.get('teacher', self.teacher)
            self.teacher = ', '.join(teacher) if isinstance(teacher, list) else teacher
            self.building = data.get('building', self.building)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def delete(self):
        """
        Remove the school class from the database and commit the transaction.
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
        Restore school classes from a list of dictionaries.

        Args:
            data (list): A list of dictionaries containing school class data.

        Returns:
            dict: A dictionary of restored school classes keyed by subject.
        """
        restored_classes = {}
        for class_data in data:
            _ = class_data.pop('id', None)  # Remove 'id' from class_data
            subject = class_data.get("subject", None)
            school_class = SchoolClass.query.filter_by(subject=subject).first()
            if school_class:
                school_class.update(class_data)
            else:
                school_class = SchoolClass(**class_data)
                school_class.create()
            restored_classes[subject] = school_class
        return restored_classes

def initSchoolClasses():
    """
    Initialize the SchoolClass table with default data.
    """
    classes = [
        SchoolClass("Mathematics", ["Nydam", "Buehler", "Froom", "Larsen", "Hightower"], building="Science Block"),
        SchoolClass("English", ["Hall", "Darcey", "Weeg"], building="Arts Wing"),
        SchoolClass("Chemistry", ["Ozuna", "Callicott", "Millman"], building="Laboratory Building"),
        SchoolClass("Engineering", ["Mortensen", "Brown", "Campillo"], building="Tech Center"),
        SchoolClass("Physics", ["Liao", "Millman", "Eckman"], building="Physics Block")
    ]

    for school_class in classes:
        try:
            db.session.add(school_class)
            db.session.commit()
            print(f"Added School Class: {school_class.subject} taught by {school_class.teacher} in {school_class.building}")
        except Exception as e:
            db.session.rollback()
            print(f"Error adding class {school_class.subject}: {e}")
