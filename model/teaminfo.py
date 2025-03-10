from __init__ import db

class TeamMember(db.Model):
    """
    TeamMember Model

    This model represents a member of the team with their personal details and the cars they own.
    """
    __tablename__ = 'team_members'

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    dob = db.Column(db.String(20), nullable=False)
    residence = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    owns_cars = db.Column(db.String(500), nullable=True)  # Comma-separated list of owned cars

    def __init__(self, first_name, last_name, dob, residence, email, owns_cars):
        self.first_name = first_name
        self.last_name = last_name
        self.dob = dob
        self.residence = residence
        self.email = email
        self.owns_cars = ', '.join(owns_cars) if isinstance(owns_cars, list) else owns_cars

    def create(self):
        """
        Add the team member to the database and commit the transaction.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def read(self):
        """
        Retrieve the team member's data as a dictionary.
        """
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "dob": self.dob,
            "residence": self.residence,
            "email": self.email,
            "owns_cars": self.owns_cars.split(', ') if self.owns_cars else []
        }

    def update(self, data):
        """
        Update the team member's data with the provided dictionary.
        """
        try:
            self.first_name = data.get('first_name', self.first_name)
            self.last_name = data.get('last_name', self.last_name)
            self.dob = data.get('dob', self.dob)
            self.residence = data.get('residence', self.residence)
            self.email = data.get('email', self.email)
            owns_cars = data.get('owns_cars', [])
            self.owns_cars = ', '.join(owns_cars) if isinstance(owns_cars, list) else owns_cars
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        
    def delete(self):
        """
        Remove the team member from the database and commit the transaction.
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
        Restore members from a list of dictionaries.

        Args:
            data (list): List of dictionaries containing vote data.
        
        Returns:
            list: List of restored Vote objects.
        """
        restored_members = {}
        for team_member in data:
            team_member_id = team_member.pop('id', None)
            member = TeamMember.query.filter_by(email=team_member['email']).first()
            if member:
                member.update(team_member)
                restored_members[team_member_id] = member
        return restored_members

def initTeamMembers():
    """
    Initialize the TeamMember table with default data.
    """
    members = [
        TeamMember("Yash", "Parikh", "July 31", "Antartica", "yashp51875@stu.powayusd.com", ["2024-McLaren-W1-HotWheels"]),
        TeamMember("Manas", "Goel", "July 12", "San Diego", "manasg67038@stu.powayusd.com", ["2024-Tesla", "2024-Mercedes"]),
        TeamMember("Mihir", "Bapat", "May 26", "Shrewsbury, UK", "mih@rb59967stu.powayusd.com", ["2022 Tesla Model Y"]),
        TeamMember("Adi", "Katre", "January 19", "La La Land", "adityak21664@stu.powayusd.com", ["2022 Tesla Model Y", "2018 BMW 328i"]),
        TeamMember("Anvay", "Vahia", "January 29", "North Pole", "anvayv22800@stu.powayusd.com", ["2023 Tesla Model Y", "2022 Hyundai Palisade"]),
        TeamMember("Pranav", "Santhosh", "May 12", "California", "pranavs22638@stu.powayusd.com", ["2023 Rivian SUV", "2024 Toyota Prius"]),
    ]

    for member in members:
        try:
            db.session.add(member)
            db.session.commit()
            print(f"Added Team Member: {member.first_name} {member.last_name}")
        except Exception as e:
            db.session.rollback()
            print(f"Error adding {member.first_name} {member.last_name}: {e}")
