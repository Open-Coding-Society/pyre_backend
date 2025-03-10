from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from api.jwt_authorize import token_required
from model.chat import Chat

chat_api = Blueprint('chat_api', __name__, url_prefix='/api')
api = Api(chat_api)

class ChatAPI:
    """
    Define the API CRUD endpoints for the Chat model.
    """
    class _CRUD(Resource):
        @token_required()
        def post(self):
            """
            Create a new chat message.
            """
            current_user = g.current_user
            data = request.get_json()

            # Validate the presence of required fields
            if not data or 'message' not in data or 'channel_id' not in data:
                return {'message': 'Message and Channel ID are required'}, 400

            # Create a new chat object
            chat = Chat(message=data['message'], user_id=current_user.id, channel_id=data['channel_id'])
            chat.create()
            return jsonify(chat.read())

        @token_required()
        def get(self):
            """
            Retrieve all chat messages by channel ID.
            """
            # Extract channel_id from query parameters
            channel_id = request.args.get('id')
            if not channel_id:
                return {'message': 'Channel ID is required'}, 400

            # Query all chat messages for the given channel_id
            chats = Chat.query.filter_by(_channel_id=channel_id).all()
            if not chats:
                return {'message': 'No chat messages found for this channel'}, 404

            # Return the list of chats in JSON format
            return jsonify([chat.read() for chat in chats])


        @token_required()
        def put(self):
            """
            Update a chat message.
            """
            current_user = g.current_user
            data = request.get_json()

            # Validate the presence of required fields
            if 'id' not in data:
                return {'message': 'Chat ID is required'}, 400
            if 'message' not in data:
                return {'message': 'Chat message is required'}, 400

            # Find the chat message by ID
            chat = Chat.query.get(data['id'])
            if not chat:
                return {'message': 'Chat message not found'}, 404

            # Update the chat message using the model's update method
            updated_chat = chat.update({'message': data['message']})
            if updated_chat:
                return jsonify(updated_chat.read())
            else:
                return {'message': 'Failed to update chat message'}, 500

        @token_required()
        def delete(self):
            """
            Delete a chat message by ID.
            """
            data = request.get_json()
            if 'id' not in data:
                return {'message': 'Chat ID is required'}, 400

            # Find the chat message by ID
            chat = Chat.query.get(data['id'])
            if not chat:
                return {'message': 'Chat message not found'}, 404

            # Delete the chat message
            chat.delete()
            return {'message': 'Chat message deleted'}, 200

    class _CHANNEL(Resource):
        @token_required()
        def post(self):
            """
            Retrieve all chat messages by channel ID.
            """
            data = request.get_json()
            if 'channel_id' not in data:
                return {'message': 'Channel ID is required'}, 400

            # Retrieve all chat messages for the given channel
            chats = Chat.query.filter_by(_channel_id=data['channel_id']).all()
            return jsonify([chat.read() for chat in chats])

    class _FILTER(Resource):
        @token_required()
        def post(self):
            """
            Retrieve all chat messages by channel ID.
            """
            data = request.get_json()
            if data is None:
                return {'message': 'Channel data not found'}, 400
            if 'channel_id' not in data:
                return {'message': 'Channel ID not found'}, 400

            # Retrieve all chat messages for the given channel
            chats = Chat.query.filter_by(_channel_id=data['channel_id']).all()
            return jsonify([chat.read() for chat in chats])

    # Add resource endpoints
    api.add_resource(_CRUD, '/chat')
    api.add_resource(_CHANNEL, '/chats/channel')
    api.add_resource(_FILTER, '/chats/filter')
