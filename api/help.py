from flask import Blueprint, request, jsonify, g, current_app
from flask_restful import Api, Resource
from api.jwt_authorize import token_required
from model.help_request import HelpRequest
from __init__ import db

help_api = Blueprint('help_api', __name__, url_prefix='/api')
api = Api(help_api)

class HelpRequestAPI:
    class _HelpRequest(Resource):
        """
        API operations for managing help requests.
        """

        @token_required()
        def post(self):
            """
            Create a new help request.
            """
            current_user = g.current_user
            data = request.get_json()

            if not data or 'message' not in data:
                return {'message': 'Message is required'}, 400

            help_request = HelpRequest(
                message=data['message'],
                user_id=current_user.id)
            try:
                help_request.create()
                return jsonify({'message': 'Help request created successfully',
                               'help_request': help_request.read()}), 201
            except Exception as e:
                current_app.logger.error(f"Error creating help request: {e}")
                return {'message': 'Failed to create help request', 'error': str(e)}, 500

        @token_required()
        def put(self):
            """
            Update a help request.
            """
            current_user = g.current_user
            data = request.get_json()

            if 'id' not in data:
                return {'message': 'Help request ID is required'}, 400
            if 'message' not in data:
                return {'message': 'Help request message is required'}, 400

            help_request = HelpRequest.query.get(data['id'])
            if not help_request:
                return {'message': 'Help request not found'}, 404

            try:
                updated_help_request = help_request.update(
                    {'message': data['message'], 'response': data.get('response'), 'status': data.get('status')})
                return jsonify({'message': 'Help request updated successfully',
                               'help_request': updated_help_request.read()})
            except Exception as e:
                current_app.logger.error(f"Error updating help request: {e}")
                return {'message': 'Failed to update help request', 'error': str(e)}, 500

        @token_required()
        def get(self):
            """
            Retrieve all help requests.
            """
            try:
                current_user = g.current_user
                help_requests = HelpRequest.query.filter_by(
                    user_id=current_user.id).all()

                if not help_requests:
                    return jsonify({'message': 'No help requests found'}), 404

                return jsonify([help_request.read()
                               for help_request in help_requests])
            except Exception as e:
                current_app.logger.error(f"Error fetching help requests: {e}")
                return jsonify({'message': 'Error fetching help requests', 'error': str(e)}), 500

        @token_required()
        def delete(self):
            """
            Delete a help request by ID.
            """
            data = request.get_json()
            if 'id' not in data:
                return {'message': 'Help request ID is required'}, 400

            help_request = HelpRequest.query.get(data['id'])
            if not help_request:
                return {'message': 'Help request not found'}, 404

            try:
                help_request.delete()
                return {'message': 'Help request deleted'}, 200
            except Exception as e:
                current_app.logger.error(f"Error deleting help request: {e}")
                return {'message': 'Failed to delete help request', 'error': str(e)}, 500

# Add resource endpoints
api.add_resource(HelpRequestAPI._HelpRequest, '/help_requests')