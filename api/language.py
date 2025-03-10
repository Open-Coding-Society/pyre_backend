import jwt
from flask import Blueprint, request, jsonify, current_app, Response, g
from flask_restful import Api, Resource  # used for REST API building
from __init__ import app
from api.jwt_authorize import token_required
from model.language import Language

# Create a Blueprint for the language API
language_api = Blueprint('language_api', __name__, url_prefix='/api')

# Create an Api object and associate it with the Blueprint
api = Api(language_api)

class LanguageAPI:
    class _Language(Resource):
        """
        API operations for managing language entries.
        """

        @token_required()
        def post(self):
            """
            Add a new language entry.
            """
            body = request.get_json()

            # Validate required fields
            name = body.get('name')
            creator = body.get('creator')
            popularity = body.get('popularity', 0)  # Default popularity is 0

            if not name or not creator:
                return {'message': 'Name and creator are required'}, 400

            try:
                # Create a new language entry
                new_language = Language(name=name, creator=creator, popularity=popularity)
                new_language.create()
                return jsonify({'message': 'Language added successfully', 'language': new_language.read()})
            except Exception as e:
                return {'message': 'Failed to create language', 'error': str(e)}, 500

        @token_required()
        def put(self):
            """
            Update an existing language entry.
            """
            body = request.get_json()

            # Validate required fields
            language_id = body.get('id')
            if not language_id:
                return {'message': 'ID is required for updating a language'}, 400

            language = Language.query.get(language_id)
            if not language:
                return {'message': 'Language not found'}, 404

            try:
                language.name = body.get('name', language.name)
                language.creator = body.get('creator', language.creator)
                language.popularity = body.get('popularity', language.popularity)
                language.create()
                return jsonify({'message': 'Language updated successfully', 'language': language.read()})
            except Exception as e:
                return {'message': 'Failed to update language', 'error': str(e)}, 500

        @token_required()
        def get(self):
            """
            Get all language entries.
            """
            try:
                languages = Language.query.all()
                return jsonify([language.read() for language in languages])
            except Exception as e:
                return {'message': 'Failed to retrieve languages', 'error': str(e)}, 500

        @token_required()
        def delete(self):
            """
            Delete an existing language entry.
            """
            body = request.get_json()

            # Validate required fields
            language_id = body.get('id')
            if not language_id:
                return {'message': 'ID is required for deleting a language'}, 400

            language = Language.query.get(language_id)
            if not language:
                return {'message': 'Language not found'}, 404

            try:
                language.delete()
                return jsonify({'message': 'Language deleted successfully'})
            except Exception as e:
                return {'message': 'Failed to delete language', 'error': str(e)}, 500

    class _Popularity(Resource):
        """
        API operations for managing language popularity.
        """

        @token_required()
        def post(self):
            """
            Upvote the popularity of a language by its ID.
            """
            body = request.get_json()

            # Validate required fields
            language_id = body.get('id')
            if not language_id:
                return {'message': 'ID is required for upvoting a language'}, 400

            language = Language.query.get(language_id)
            if not language:
                return {'message': 'Language not found'}, 404

            try:
                language.upvote()
                return jsonify({'message': 'Popularity increased successfully', 'language': language.read()})
            except Exception as e:
                return {'message': 'Failed to increase popularity', 'error': str(e)}, 500

# Register the API resources with the Blueprint
api.add_resource(LanguageAPI._Language, '/language')
api.add_resource(LanguageAPI._Popularity, '/language/popularity')