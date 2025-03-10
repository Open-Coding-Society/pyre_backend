from __init__ import app, db

class HelpRequest(db.Model):
    __tablename__ = 'help_requests'
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255), nullable=False)
    response = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(50), default='Pending')
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error creating HelpRequest: {e}")
            raise e

    def delete(self):
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error deleting HelpRequest: {e}")
            raise e

    def read(self):
        return {
            'id': self.id,
            'message': self.message,
            'response': self.response,
            'status': self.status,
            'user_id': self.user_id
        }

    def update(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        try:
            db.session.commit()
            return self
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating HelpRequest: {e}")
            raise e

def initHelpRequests():
    with app.app_context():
        db.create_all()
        tester_data = [
            HelpRequest(message='Need help with login', user_id=1),
            HelpRequest(message='Unable to access dashboard', user_id=2),
            HelpRequest(message='Error in report generation', user_id=3)
        ]
        for data in tester_data:
            try:
                db.session.add(data)
                db.session.commit()
                print(f"Record created: {repr(data)}")
            except Exception as e:
                db.session.rollback()
                print(f"Error creating record for help request {data.message}: {e}")