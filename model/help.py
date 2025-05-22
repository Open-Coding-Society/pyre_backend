"""
Help System Database Models
Defines the database structure for help requests, FAQ, responses, and history tracking.
"""

from __init__ import app, db
from datetime import datetime
from sqlalchemy.exc import IntegrityError

class HelpRequest(db.Model):
    """
    HelpRequest model for storing user help requests.
    """
    __tablename__ = 'help_requests'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    priority = db.Column(db.String(20), nullable=False, default='medium')
    status = db.Column(db.String(20), nullable=False, default='open')
    contact_email = db.Column(db.String(100))
    admin_notes = db.Column(db.Text)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    assigned_to_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.now)
    updated_date = db.Column(db.DateTime, default=datetime.now)
    resolved_date = db.Column(db.DateTime)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='help_requests')
    assigned_to = db.relationship('User', foreign_keys=[assigned_to_id], backref='assigned_help_requests')
    responses = db.relationship('HelpResponse', backref='help_request', cascade='all, delete-orphan')
    history = db.relationship('HelpHistory', backref='help_request', cascade='all, delete-orphan')

    def __init__(self, title, description, category, priority, user_id, contact_email=None, status='open'):
        self.title = title
        self.description = description
        self.category = category
        self.priority = priority
        self.user_id = user_id
        self.contact_email = contact_email
        self.status = status

    def __repr__(self):
        return f"<HelpRequest {self.id}: {self.title}>"

    def create(self):
        """Create a new help request in the database."""
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        """Return a dictionary representation of the help request."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "status": self.status,
            "contact_email": self.contact_email,
            "admin_notes": self.admin_notes,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else None,
            "assigned_to_id": self.assigned_to_id,
            "assigned_to_name": self.assigned_to.name if self.assigned_to else None,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None,
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "response_count": len(self.responses) if self.responses else 0
        }

    def update(self):
        """Update the help request in the database."""
        try:
            self.updated_date = datetime.now()
            if self.status in ['resolved', 'closed'] and not self.resolved_date:
                self.resolved_date = datetime.now()
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def delete(self):
        """Delete the help request from the database."""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except IntegrityError:
            db.session.rollback()
            return False

    @staticmethod
    def restore(data_list):
        """Restore help requests from a list of dictionaries."""
        for data in data_list:
            help_request = HelpRequest.query.get(data['id'])
            if help_request is None:
                help_request = HelpRequest(
                    title=data['title'],
                    description=data['description'],
                    category=data['category'],
                    priority=data['priority'],
                    user_id=data['user_id'],
                    contact_email=data.get('contact_email'),
                    status=data.get('status', 'open')
                )
                help_request.id = data['id']
                help_request.create()


class HelpResponse(db.Model):
    """
    HelpResponse model for storing admin responses to help requests.
    """
    __tablename__ = 'help_responses'

    id = db.Column(db.Integer, primary_key=True)
    response_text = db.Column(db.Text, nullable=False)
    is_public = db.Column(db.Boolean, default=True)
    is_solution = db.Column(db.Boolean, default=False)
    
    # Foreign keys
    help_request_id = db.Column(db.Integer, db.ForeignKey('help_requests.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.now)
    updated_date = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    admin = db.relationship('User', backref='help_responses')

    def __init__(self, help_request_id, admin_id, response_text, is_public=True, is_solution=False):
        self.help_request_id = help_request_id
        self.admin_id = admin_id
        self.response_text = response_text
        self.is_public = is_public
        self.is_solution = is_solution

    def __repr__(self):
        return f"<HelpResponse {self.id} for Request {self.help_request_id}>"

    def create(self):
        """Create a new help response in the database."""
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        """Return a dictionary representation of the help response."""
        return {
            "id": self.id,
            "response_text": self.response_text,
            "is_public": self.is_public,
            "is_solution": self.is_solution,
            "help_request_id": self.help_request_id,
            "admin_id": self.admin_id,
            "admin_name": self.admin.name if self.admin else None,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None
        }

    def update(self):
        """Update the help response in the database."""
        try:
            self.updated_date = datetime.now()
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def delete(self):
        """Delete the help response from the database."""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except IntegrityError:
            db.session.rollback()
            return False


class HelpHistory(db.Model):
    """
    HelpHistory model for tracking all actions performed on help requests.
    """
    __tablename__ = 'help_history'

    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(50), nullable=False)  # created, updated, response_added, status_changed, etc.
    details = db.Column(db.Text)
    
    # Foreign keys
    help_request_id = db.Column(db.Integer, db.ForeignKey('help_requests.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    user = db.relationship('User', backref='help_history')

    def __init__(self, help_request_id, action, details, user_id):
        self.help_request_id = help_request_id
        self.action = action
        self.details = details
        self.user_id = user_id

    def __repr__(self):
        return f"<HelpHistory {self.id}: {self.action}>"

    def create(self):
        """Create a new help history entry in the database."""
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        """Return a dictionary representation of the help history entry."""
        return {
            "id": self.id,
            "action": self.action,
            "details": self.details,
            "help_request_id": self.help_request_id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else None,
            "created_date": self.created_date.isoformat() if self.created_date else None
        }


class FAQ(db.Model):
    """
    FAQ model for storing frequently asked questions and answers.
    """
    __tablename__ = 'faqs'

    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    tags = db.Column(db.String(255))  # Comma-separated tags
    display_order = db.Column(db.Integer, default=0)
    view_count = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    
    # Foreign keys
    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    updated_by_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.now)
    updated_date = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    created_by = db.relationship('User', foreign_keys=[created_by_id], backref='created_faqs')
    updated_by = db.relationship('User', foreign_keys=[updated_by_id], backref='updated_faqs')

    def __init__(self, question, answer, category, created_by_id, tags='', display_order=0):
        self.question = question
        self.answer = answer
        self.category = category
        self.created_by_id = created_by_id
        self.tags = tags
        self.display_order = display_order

    def __repr__(self):
        return f"<FAQ {self.id}: {self.question[:50]}...>"

    def create(self):
        """Create a new FAQ in the database."""
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        """Return a dictionary representation of the FAQ."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "tags": self.tags.split(',') if self.tags else [],
            "display_order": self.display_order,
            "view_count": self.view_count,
            "is_active": self.is_active,
            "created_by_id": self.created_by_id,
            "created_by_name": self.created_by.name if self.created_by else None,
            "updated_by_id": self.updated_by_id,
            "updated_by_name": self.updated_by.name if self.updated_by else None,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None
        }

    def update(self):
        """Update the FAQ in the database."""
        try:
            self.updated_date = datetime.now()
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def delete(self):
        """Delete the FAQ from the database."""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except IntegrityError:
            db.session.rollback()
            return False

    def increment_view_count(self):
        """Increment the view count for this FAQ."""
        try:
            self.view_count += 1
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    @staticmethod
    def restore(data_list):
        """Restore FAQs from a list of dictionaries."""
        for data in data_list:
            faq = FAQ.query.get(data['id'])
            if faq is None:
                faq = FAQ(
                    question=data['question'],
                    answer=data['answer'],
                    category=data['category'],
                    created_by_id=data['created_by_id'],
                    tags=','.join(data.get('tags', [])) if isinstance(data.get('tags'), list) else data.get('tags', ''),
                    display_order=data.get('display_order', 0)
                )
                faq.id = data['id']
                faq.create()


class HelpCategory(db.Model):
    """
    HelpCategory model for managing help request categories.
    """
    __tablename__ = 'help_categories'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    display_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    color = db.Column(db.String(7), default='#007bff')  # Hex color code
    icon = db.Column(db.String(50))  # Icon class name
    is_active = db.Column(db.Boolean, default=True)
    display_order = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_date = db.Column(db.DateTime, default=datetime.now)
    updated_date = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, name, display_name, description='', color='#007bff', icon='', display_order=0):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.color = color
        self.icon = icon
        self.display_order = display_order

    def __repr__(self):
        return f"<HelpCategory {self.name}>"

    def create(self):
        """Create a new help category in the database."""
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        """Return a dictionary representation of the help category."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "color": self.color,
            "icon": self.icon,
            "is_active": self.is_active,
            "display_order": self.display_order,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None
        }

    def update(self):
        """Update the help category in the database."""
        try:
            self.updated_date = datetime.now()
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def delete(self):
        """Delete the help category from the database."""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except IntegrityError:
            db.session.rollback()
            return False


def initHelpSystem():
    """Initialize the help system with default categories and sample data."""
    print("Initializing Help System...")
    
    # Create default categories
    categories = [
        {
            'name': 'technical',
            'display_name': 'Technical Issues',
            'description': 'Problems with system functionality, errors, or performance',
            'color': '#dc3545',
            'icon': 'fas fa-cog',
            'display_order': 1
        },
        {
            'name': 'account',
            'display_name': 'Account & Login',
            'description': 'Account access, password resets, profile issues',
            'color': '#007bff',
            'icon': 'fas fa-user',
            'display_order': 2
        },
        {
            'name': 'general',
            'display_name': 'General Questions',
            'description': 'General inquiries and questions',
            'color': '#28a745',
            'icon': 'fas fa-question-circle',
            'display_order': 3
        },
        {
            'name': 'bug_report',
            'display_name': 'Bug Reports',
            'description': 'Report software bugs and issues',
            'color': '#ffc107',
            'icon': 'fas fa-bug',
            'display_order': 4
        },
        {
            'name': 'feature_request',
            'display_name': 'Feature Requests',
            'description': 'Suggest new features or improvements',
            'color': '#17a2b8',
            'icon': 'fas fa-lightbulb',
            'display_order': 5
        },
        {
            'name': 'billing',
            'display_name': 'Billing & Payments',
            'description': 'Payment issues, billing questions, subscriptions',
            'color': '#6f42c1',
            'icon': 'fas fa-credit-card',
            'display_order': 6
        },
        {
            'name': 'other',
            'display_name': 'Other',
            'description': 'Issues that don\'t fit into other categories',
            'color': '#6c757d',
            'icon': 'fas fa-ellipsis-h',
            'display_order': 7
        }
    ]
    
    for cat_data in categories:
        category = HelpCategory.query.filter_by(name=cat_data['name']).first()
        if not category:
            category = HelpCategory(**cat_data)
            category.create()
    
    # Create sample FAQs
    from model.user import User
    admin_user = User.query.filter_by(role='Admin').first()
    if admin_user:
        sample_faqs = [
            {
                'question': 'How do I reset my password?',
                'answer': 'To reset your password:\n1. Go to the login page\n2. Click "Forgot Password?"\n3. Enter your email address\n4. Check your email for reset instructions\n5. Follow the link in the email to create a new password',
                'category': 'account',
                'tags': 'password,reset,login,account',
                'display_order': 1
            },
            {
                'question': 'How do I update my profile information?',
                'answer': 'To update your profile:\n1. Log into your account\n2. Click on your profile picture or name\n3. Select "Edit Profile"\n4. Update your information\n5. Click "Save Changes"',
                'category': 'account',
                'tags': 'profile,update,edit,information',
                'display_order': 2
            },
            {
                'question': 'What browsers are supported?',
                'answer': 'Our platform supports the following browsers:\n- Chrome (latest version)\n- Firefox (latest version)\n- Safari (latest version)\n- Edge (latest version)\n\nFor the best experience, please keep your browser updated.',
                'category': 'technical',
                'tags': 'browser,support,compatibility,technical',
                'display_order': 1
            },
            {
                'question': 'How do I report a bug?',
                'answer': 'To report a bug:\n1. Go to the Help section\n2. Click "Submit a Request"\n3. Select "Bug Report" as the category\n4. Provide detailed information about the issue\n5. Include steps to reproduce the bug\n6. Submit your report\n\nOur team will investigate and respond promptly.',
                'category': 'bug_report',
                'tags': 'bug,report,issue,problem',
                'display_order': 1
            },
            {
                'question': 'How can I suggest a new feature?',
                'answer': 'We love hearing your ideas! To suggest a new feature:\n1. Submit a help request\n2. Select "Feature Request" as the category\n3. Describe your idea in detail\n4. Explain how it would benefit users\n5. Submit your suggestion\n\nWe review all feature requests and consider them for future updates.',
                'category': 'feature_request',
                'tags': 'feature,request,suggestion,idea',
                'display_order': 1
            }
        ]
        
        for faq_data in sample_faqs:
            existing_faq = FAQ.query.filter_by(question=faq_data['question']).first()
            if not existing_faq:
                faq = FAQ(
                    question=faq_data['question'],
                    answer=faq_data['answer'],
                    category=faq_data['category'],
                    created_by_id=admin_user.id,
                    tags=faq_data['tags'],
                    display_order=faq_data['display_order']
                )
                faq.create()
    
    print("Help System initialized successfully!")