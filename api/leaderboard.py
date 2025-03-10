from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.user import User
from __init__ import db
from model.topinterests import TopInterest

# Create blueprint for the leaderboard API
leaderboard_api = Blueprint('leaderboard_api', __name__, url_prefix='/api')
api = Api(leaderboard_api)

# Create class for Leaderboard
class Leaderboard:
    # Create class for top users
    class _TopUsers(Resource):
        def get(self):
            try:
                # Get the current user ID from the request arguments
                current_user_id = request.args.get('user_id')
                
                # Query the database to get the current user by ID
                current_user = User.query.get(current_user_id)
                
                # If the current user is not found, return a 404 error message
                if not current_user:
                    return {'message': 'User not found'}, 404
                
                # Split the current user's interests into a set of interests
                current_user_interests = set(current_user._interests.split(", "))
                
                # Query the database to get all users except the current user
                all_users = User.query.filter(User.id != current_user_id).all()
                
                matched_users = []
                # Iterate over all users
                for user in all_users:
                    # Split each user's interests into a set of interests
                    user_interests = set(user._interests.split(", "))
                    
                    # Find the shared interests between the current user and each user
                    shared_interests = current_user_interests.intersection(user_interests)
                    
                    # If there are shared interests, add the user to the matched users list
                    if shared_interests:
                        matched_users.append({
                            'username': user._name,
                            'shared_interests': list(shared_interests)
                        })
                
                # Sort the matched users by the number of shared interests in descending order
                matched_users.sort(key=lambda x: len(x['shared_interests']), reverse=True)
                
                # Return the matched users as a JSON response
                return jsonify({'top_users': matched_users})
            except Exception as e:
                # If there is an error, return a 500 error message
                return {'message': f'Error retrieving top users: {str(e)}'}, 500

    # Create class for top interests
    class _TopInterests(Resource):
        def get(self):
            try:
                # Retrieve all users
                all_users = User.query.all()
                
                interest_counts = {}
                # Iterate over all users
                for user in all_users:
                    # Split each user's interests into a list of interests
                    interests = user._interests.split(", ")
                    
                    # Count the occurrences of each interest
                    for interest in interests:
                        if interest in interest_counts:
                            interest_counts[interest] += 1
                        else:
                            interest_counts[interest] = 1
                
                # Sort interests by count in descending order
                sorted_interests = sorted(interest_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Return the top interests as a JSON response
                return jsonify({'top_interests': [{'interest': interest, 'count': count} for interest, count in sorted_interests]})
            except Exception as e:
                # If there is an error, return a 500 error message
                return {'message': f'Error retrieving top interests: {str(e)}'}, 500

    # Create class for CRUD operations on top interests
    class _CRUD(Resource):
        def post(self):
            """
            Create a new top interest.
            """
            data = request.get_json()

            if not data or '_interests' not in data or 'count' not in data:
                return {'message': 'Interest and count are required'}, 400

            top_interest = TopInterest(
                _interests=data['_interests'],
                count=data['count']
            )

            try:
                top_interest.create()
                return jsonify(top_interest.read())
            except Exception as e:
                return {'message': f'Error creating top interest: {e}'}, 500

        def get(self):
            """
            Retrieve all top interests.
            """
            top_interests = TopInterest.query.all()
            return jsonify([top_interest.read() for top_interest in top_interests])

        def put(self):
            """
            Update an existing top interest.
            """
            data = request.get_json()

            if not data or '_interests' not in data:
                return {'message': 'Interest is required for updating a top interest'}, 400

            top_interest = TopInterest.query.filter_by(_interests=data['_interests']).first()
            if not top_interest:
                return {'message': 'Top interest not found'}, 404

            try:
                top_interest.update(data)
                return jsonify(top_interest.read())
            except Exception as e:
                return {'message': f'Error updating top interest: {e}'}, 500

        def delete(self):
            """
            Delete a top interest.
            """
            data = request.get_json()

            if not data or '_interests' not in data:
                return {'message': 'Interest is required for deleting a top interest'}, 400

            top_interest = TopInterest.query.filter_by(_interests=data['_interests']).first()
            if not top_interest:
                return {'message': 'Top interest not found'}, 404

            try:
                top_interest.delete()
                return {'message': 'Top interest deleted successfully'}, 200
            except Exception as e:
                return {'message': f'Error deleting top interest: {e}'}, 500

# Add resources to the API
api.add_resource(Leaderboard._TopUsers, '/leaderboard/top_users')
api.add_resource(Leaderboard._TopInterests, '/leaderboard/top_interests')
api.add_resource(Leaderboard._CRUD, '/leaderboard/top_interests/crud')