from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.usettings import Settings
from api.jwt_authorize import token_required

# Create a Blueprint for the settings API
settings_api = Blueprint('settings_api', __name__, url_prefix='/api')

# Create an Api object and associate it with the Blueprint
api = Api(settings_api)

class SettingsAPI(Resource):
    @token_required()
    def get(self):
        """
        Get the current settings.
        """
        settings = Settings.query.first()
        if settings:
            return jsonify({
                'description': settings.description,
                'contact_email': settings.contact_email,
                'contact_phone': settings.contact_phone
            })
        return jsonify({'message': 'Settings not found'}), 404

    @token_required()
    def post(self):
        """
        Create new settings.
        """
        data = request.get_json()
        settings = Settings(
            description=data.get('description'),
            contact_email=data.get('contact_email'),
            contact_phone=data.get('contact_phone')
        )
        db.session.add(settings)
        db.session.commit()
        return jsonify({'message': 'Settings created successfully'}), 201

    @token_required()
    def put(self):
        """
        Update the settings.
        """
        data = request.get_json()
        settings = Settings.query.first()
        if settings:
            settings.description = data.get('description', settings.description)
            settings.contact_email = data.get('contact_email', settings.contact_email)
            settings.contact_phone = data.get('contact_phone', settings.contact_phone)
            db.session.commit()
            return jsonify({'message': 'Settings updated successfully'})
        return jsonify({'message': 'Settings not found'}), 404

    @token_required()
    def read(self):
        """
        Read the settings.
        """
        settings = Settings.query.first()
        if settings:
            return jsonify({
                'description': settings.description,
                'contact_email': settings.contact_email,
                'contact_phone': settings.contact_phone
            })
        return jsonify({'message': 'Settings not found'}), 404

# Add resources to the API
api.add_resource(SettingsAPI, '/settings')