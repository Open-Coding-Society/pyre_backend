from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource
from flask_login import login_required
from model.poll import Poll

poll_met_api = Blueprint('poll_met_api', __name__, url_prefix='/api')
api = Api(poll_met_api)

class PollAPI:
    """
    Define the API CRUD endpoints for the Poll model.
    """
    
    class _CRUD(Resource):
        @login_required
        def post(self):
            """
            Create a new poll.
            """
            data = request.get_json()
            if not data or 'name' not in data or 'interests' not in data:
                return {'message': 'Name and interests fields are required.'}, 400
            
            poll = Poll(name=data['name'], interests=data['interests'])
            poll.create()
            return jsonify(poll.read())

        @login_required
        def get(self):
            """
            Retrieve all polls.
            """
            polls = Poll.query.all()
            if not polls:
                return {'message': 'No polls found'}, 404

            return jsonify([poll.read() for poll in polls])

        @login_required
        def put(self):
            """
            Update an existing poll.
            """
            data = request.get_json()
            if 'id' not in data:
                return {'message': 'Poll ID is required.'}, 400
            
            poll = Poll.query.get(data['id'])
            if not poll:
                return {'message': 'Poll not found.'}, 404
            
            if 'name' in data:
                poll.name = data['name']
            if 'interests' in data:
                poll.interests = data['interests']
            
            updated_poll = poll.update({"name": poll.name, "interests": poll.interests})
            if updated_poll:
                return jsonify(updated_poll.read())
            else:
                return {'message': 'Failed to update poll'}, 500

        @login_required
        def delete(self):
            """
            Delete a poll.
            """
            data = request.get_json()
            if 'id' not in data:
                return {'message': 'Poll ID is required.'}, 400
            
            poll = Poll.query.get(data['id'])
            if not poll:
                return {'message': 'Poll not found.'}, 404

            poll.delete()
            return {'message': 'Poll deleted successfully'}, 200

    class _FILTER(Resource):
        @login_required
        def post(self):
            """
            Retrieve all polls filtered by name.
            """
            data = request.get_json()
            if 'name' not in data:
                return {'message': 'Poll name is required for filtering.'}, 400

            polls = Poll.query.filter_by(name=data['name']).all()
            if not polls:
                return {'message': 'No polls found with the specified name.'}, 404

            return jsonify([poll.read() for poll in polls])

    # Add resource endpoints
    api.add_resource(_CRUD, '/pollmet')
    api.add_resource(_FILTER, '/pollsmet/filter')
