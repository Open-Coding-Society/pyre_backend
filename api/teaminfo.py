from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.teaminfo import TeamMember

# Define the Blueprint for the TeamMember API
team_member_api = Blueprint('TeamMember_api', __name__, url_prefix='/api')

# Connect the Api object to the Blueprint
api = Api(team_member_api)

class TeamMemberAPI:
    """
    Define the API CRUD endpoints for the TeamMember model.
    """

    class _CRUD(Resource):
        def post(self):
            """
            Create a new team member.
            """
            data = request.get_json()

            # Validate required fields
            if not data or 'first_name' not in data or 'last_name' not in data or 'dob' not in data or 'residence' not in data or 'email' not in data or 'owns_cars' not in data:
                return {'message': 'First name, last name, date of birth, residence, email, and car ownership are required'}, 400

            # Create a new teammember object
            team_member = TeamMember(
                first_name=data['first_name'],
                last_name=data['last_name'],
                dob=data['dob'],
                residence=data['residence'],
                email=data['email'],
                owns_cars=data['owns_cars'] if isinstance(data['owns_cars'], list) else [data['owns_cars']]
            )

            # Save the team member using the ORM method
            try:
                team_member.create()
                return jsonify(team_member.read())
            except Exception as e:
                return {'message': f'Error creating team member: {e}'}, 500

        def get(self):
            """
            Retrieve all team members by ID.
            """
            member_id = request.args.get('id')

            if member_id:
                # Get a specific team member by ID
                team_member = TeamMember.query.get(member_id)
                if not team_member:
                    return {'message': 'Team member not found'}, 404
                return jsonify(team_member.read())
            
            # Get all team members
            members = TeamMember.query.all()
            return jsonify([team_member.read() for team_member in members])
        

        def put(self):
            """
            Update an existing team member by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for updating a team member'}, 400

            # Find the team member by ID
            team_member = TeamMember.query.get(data['id'])
            if not team_member:
                return {'message': 'Team member not found'}, 404

            # Update the team member
            try:
                team_member.update(data)
                return jsonify(team_member.read())
            except Exception as e:
                return {'message': f'Error updating team member: {e}'}, 500

        def delete(self):
            """
            Delete a team member by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for deleting a member'}, 400

            # Find the team member by ID
            team_member = TeamMember.query.get(data['id'])
            if not team_member:
                return {'message': 'Team member not found'}, 404

            # Delete the team member
            try:
                team_member.delete()
                return {'message': 'Team Member deleted successfully'}, 200
            except Exception as e:
                return {'message': f'Error deleting team member: {e}'}, 500

    class _BY_CAR(Resource):
        def get(self):
            """
            Retrieve all team members who own a specific car.
            """
            car = request.args.get('car')

            if not car:
                return {'message': 'Car is required'}, 400

            # Get all team members who own the car
            members = TeamMember.query.filter(TeamMember.owns_cars.contains(car)).all()
            return jsonify([team_member.read() for team_member in members])

    # Map the resources to API endpoints
    api.add_resource(_CRUD, '/team_member')
    api.add_resource(_BY_CAR, '/team_member/car')