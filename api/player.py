from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.player import Player

# Define the Blueprint for the Player API
player_api = Blueprint('player_api', __name__, url_prefix='/api')

# Connect the Api object to the Blueprint
api = Api(player_api)

class PlayerAPI:
    """
    Define the API CRUD endpoints for the Player model.
    """

    class _CRUD(Resource):
        def post(self):
            """
            Create a new player.
            """
            data = request.get_json()

            # Validate required fields
            if not data or 'first_name' not in data or 'last_name' not in data or 'team' not in data:
                return {'message': 'First name, last name, and team are required'}, 400

            # Create a new Player object
            player = Player(
                first_name=data['first_name'],
                last_name=data['last_name'],
                dob=data.get('dob', ''),
                residence=data.get('residence', ''),
                email=data.get('email', ''),
                team=data['team'],
                sports_played=data.get('sports_played', [])
            )

            # Save the player using the ORM method
            try:
                player.create()
                return jsonify(player.read())
            except Exception as e:
                return {'message': f'Error creating player: {e}'}, 500

        def get(self):
            """
            Retrieve all players or a specific player by ID.
            """
            player_id = request.args.get('id')

            if player_id:
                # Get a specific player by ID
                player = Player.query.get(player_id)
                if not player:
                    return {'message': 'Player not found'}, 404
                return jsonify(player.read())

            # Get all players
            players = Player.query.all()
            return jsonify([player.read() for player in players])

        def put(self):
            """
            Update an existing player by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for updating a player'}, 400

            # Find the player by ID
            player = Player.query.get(data['id'])
            if not player:
                return {'message': 'Player not found'}, 404

            # Update the player
            try:
                for key, value in data.items():
                    if hasattr(player, key):
                        setattr(player, key, value)
                db.session.commit()
                return jsonify(player.read())
            except Exception as e:
                return {'message': f'Error updating player: {e}'}, 500

        def delete(self):
            """
            Delete a player by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for deleting a player'}, 400

            # Find the player by ID
            player = Player.query.get(data['id'])
            if not player:
                return {'message': 'Player not found'}, 404

            # Delete the player
            try:
                player.delete()
                return {'message': 'Player deleted successfully'}, 200
            except Exception as e:
                return {'message': f'Error deleting player: {e}'}, 500

    class _BY_TEAM(Resource):
        def get(self):
            """
            Retrieve all players from a specific team.
            """
            team_name = request.args.get('team')

            if not team_name:
                return {'message': 'Team name is required'}, 400

            players = Player.query.filter(Player.team.like(f"%{team_name}%")).all()
            if not players:
                return {'message': 'No players found for the specified team'}, 404

            return jsonify([player.read() for player in players])

    # Map the resources to API endpoints
    api.add_resource(_CRUD, '/player')
    api.add_resource(_BY_TEAM, '/player/team')
