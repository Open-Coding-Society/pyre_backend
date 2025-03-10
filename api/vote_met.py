import jwt
from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource  # used for REST API building
from datetime import datetime
from __init__ import app
from flask_login import login_required
from model.post import Post
from model.vote import Vote

# Define the Blueprint for the Vote API
vote_met_api = Blueprint('vote_met_api', __name__, url_prefix='/api')

# Connect the Api object to the Blueprint
api = Api(vote_met_api)

class VoteAPI:
    """
    Define the API CRUD endpoints for the Vote model.
    There are operations for upvoting, downvoting, and retrieving votes for a post.
    """

    class _CRUD(Resource):
        @login_required
        def post(self):
            """
            Create or update a vote (upvote or downvote) for a post.
            """
            # Get current user from the token
            current_user = g.current_user
            # Get the request data
            data = request.get_json()

            # Validate required fields
            if not data:
                return {'message': 'No input data provided'}, 400
            if 'post_id' not in data:
                return {'message': 'Post ID is required'}, 400
            if 'vote_type' not in data or data['vote_type'] not in ['upvote', 'downvote']:
                return {'message': 'Vote type must be "upvote" or "downvote"'}, 400

            # Check if the vote already exists for the user on the post
            existing_vote = Vote.query.filter_by(_post_id=data['post_id'], _user_id=current_user.id).first()
            if existing_vote:
                # Update the existing vote type
                existing_vote._vote_type = data['vote_type']
                existing_vote.create()  # This will commit the update
                return jsonify(existing_vote.read())

            # Create a new vote object
            vote = Vote(data['vote_type'], current_user.id, data['post_id'])
            # Save the vote using the ORM method
            vote.create()
            # Return the saved vote in JSON format
            return jsonify(vote.read())

        @login_required
        def put(self):
            """
            Update the vote type for a specific vote by ID.
            """
            # Get current user from the token
            current_user = g.current_user
            # Get the request data
            data = request.get_json()

            # Validate required fields
            if not data:
                return {'message': 'No input data provided'}, 400
            if 'post_id' not in data:
                return {'message': 'Post ID is required'}, 400
            if 'vote_type' not in data or data['vote_type'] not in ['upvote', 'downvote']:
                return {'message': 'Vote type must be "upvote" or "downvote"'}, 400

            # Find the vote by ID and ensure it belongs to the current user
            vote = Vote.query.filter_by(_post_id=data['post_id'], _user_id=current_user.id).first()
            if not vote:
                return {'message': 'Vote not found or not authorized to update this vote'}, 404

            # Update the vote type
            vote.update(data['vote_type'])
            return jsonify(vote.read())

        @login_required
        def delete(self):
            """
            Remove a vote by a user on a specific post.
            """
            # Get current user from the token
            current_user = g.current_user
            # Get the request data
            data = request.get_json()

            # Validate required fields
            if not data or 'post_id' not in data:
                return {'message': 'Post ID is required'}, 400

            # Find the vote by user and post
            vote = Vote.query.filter_by(_post_id=data['post_id'], _user_id=current_user.id).first()
            if vote is None:
                return {'message': 'Vote not found'}, 404

            # Delete the vote
            vote.delete()
            return jsonify({"message": "Vote removed"})

    class _POST_VOTES(Resource):
        def get(self):
            """
            Retrieve all votes for a specific post, including counts of upvotes and downvotes.
            """
            # Get post_id from query parameters
            post_id = request.args.get('post_id')
            
            if not post_id:
                return {'message': 'Post ID is required as a query parameter'}, 400

            # Get all votes for the post
            votes = Vote.query.filter_by(_post_id=post_id).all()
            upvotes = [vote.read() for vote in votes if vote._vote_type == 'upvote']
            downvotes = [vote.read() for vote in votes if vote._vote_type == 'downvote']

            result = {
                "post_id": post_id,
                "upvote_count": len(upvotes),
                "downvote_count": len(downvotes),
                "total_votes": len(votes),
                "upvotes": upvotes,
                "downvotes": downvotes
            }
            return jsonify(result)

    """
    Map the _CRUD and _POST_VOTES classes to the API endpoints for /vote and /vote/post.
    - The _CRUD class defines the HTTP methods for voting (post and delete).
    - The _POST_VOTES class defines the endpoint for retrieving all votes for a specific post.
    """
    api.add_resource(_CRUD, '/votemet')
    api.add_resource(_POST_VOTES, '/votemet/post')
