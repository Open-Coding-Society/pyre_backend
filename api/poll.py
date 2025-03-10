from api.jwt_authorize import token_required
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.poll import Poll

# Blueprint for Poll API
poll_api = Blueprint('poll_api', __name__, url_prefix='/api')
api = Api(poll_api)


class PollAPI:
    """
    Define the API endpoints for the Poll model.
    """

    class _Read(Resource): # R = Read
        """
        GET request handler: Read all polls.
        """
        @token_required()
        def get(self):
            try:
                # Retrieve all poll records
                polls = Poll.query.all()
                poll_list = []
                for poll in polls:
                    poll_list.append(poll.read())
                return jsonify(poll_list)
            except Exception as e:
                print(f"Poll Read Error: {e}")
                return {'message': f'Error retrieving poll data: {str(e)}'}, 500

    class _Create(Resource): # C = Create
        """
        POST request handler: Create a new poll.
        """
        @token_required()
        def post(self):
            try:
                data = request.get_json()
                if not data:
                    return {'message': 'No input data provided'}, 400

                name = data.get('name')
                interests = data.get('interests')

                # Basic validation
                if not name or interests is None:
                    return {'message': 'name and interests fields are required.'}, 422

                # Create and save the new Poll
                new_poll = Poll(name, interests)
                new_poll.create()

                return {'message': 'Poll data inserted successfully'}, 201

            except KeyError as e:
                return {'message': f'Missing field: {str(e)}'}, 400
            except Exception as e:
                print(f"Poll Create Error: {e}")
                return {'message': f'Error inserting poll data: {str(e)}'}, 500

    class _Update(Resource): # U = Update
        @token_required()
        def put(self):
            try:
                data = request.get_json()
                if not data:
                    return {'message': 'No input data provided'}, 400

                poll_id = data.get('id')
                if not poll_id:
                    return {'message': 'Poll ID is required.'}, 400

                poll = Poll.query.get(poll_id)
                if not poll:
                    return {'message': 'Poll not found.'}, 404

                name = data.get('name')
                interests = data.get('interests')

                if name:
                    poll.name = name
                else:
                    poll.name = poll.name

                if interests is not None:
                    poll.interests = interests
                else:
                    poll.interests = poll.interests

                poll.update({
                    "name": poll.name,
                    "interests": poll.interests
                })
                return {'message': 'Poll updated successfully'}, 200

            except KeyError as e:
                return {'message': f'Missing field: {str(e)}'}, 400
            except Exception as e:
                print(f"Poll Update Error: {e}")
                return {'message': f'Error updating poll: {str(e)}'}, 500

    class _Delete(Resource): # D = Delete
        @token_required()
        def delete(self):
            try:
                data = request.get_json()
                if not data:
                    return {'message': 'No input data provided'}, 400

                poll_id = data.get('id')
                if not poll_id:
                    return {'message': 'Poll ID is required.'}, 422

                poll = Poll.query.get(poll_id)
                if not poll:
                    return {'message': 'Poll not found.'}, 404

                poll.delete()
                return {'message': 'Poll deleted successfully'}, 200

            except KeyError as e:
                return {'message': f'Missing field: {str(e)}'}, 400
            except Exception as e:
                print(f"Poll Delete Error: {e}")
                return {'message': f'Error deleting poll: {str(e)}'}, 500

# Map the resources to their endpoints
api.add_resource(PollAPI._Read, '/poll')   # GET -> read all polls
api.add_resource(PollAPI._Create, '/poll')  # POST -> create new poll
api.add_resource(PollAPI._Update, '/poll')  # PUT -> update existing poll
api.add_resource(PollAPI._Delete, '/poll')  # DELETE -> delete existing poll
