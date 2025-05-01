from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
from model.fire import ForestFireModel, ForestFire
from __init__ import app, db

forest_fire_api = Blueprint('forest_fire_api', __name__,
                   url_prefix='/api/forest-fire')

api = Api(forest_fire_api)

class ForestFireAPI:
    class _Predict(Resource):
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending forest fire data to the server to get a prediction fits the semantics of a POST request.

            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """
            # Get the forest fire data from the request
            forest_data = request.get_json()

            # Validate the forest fire data
            required_fields = ['X', 'Y', 'month', 'day', 'temp', 'RH', 'wind', 'rain']
            for field in required_fields:
                if field not in forest_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Get the singleton instance of the ForestFireModel
            forestFireModel = ForestFireModel.get_instance()
            # Predict the burned area
            response = forestFireModel.predict(forest_data)

            # Return the response as JSON
            return jsonify(response)

    class _Create(Resource):
        def post(self):
            """Create a new forest fire record"""
            # Get data from request body
            forest_data = request.get_json()

            # Validate the forest fire data
            required_fields = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
            for field in required_fields:
                if field not in forest_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Create new forest fire instance
            try:
                forest_fire = ForestFire(
                    X=forest_data['X'],
                    Y=forest_data['Y'],
                    month=forest_data['month'],
                    day=forest_data['day'],
                    FFMC=forest_data['FFMC'],
                    DMC=forest_data['DMC'],
                    DC=forest_data['DC'],
                    ISI=forest_data['ISI'],
                    temp=forest_data['temp'],
                    RH=forest_data['RH'],
                    wind=forest_data['wind'],
                    rain=forest_data['rain'],
                    area=forest_data['area']
                )
                forest_fire.create()
                return {"message": "Forest fire record created successfully", "forest_fire": forest_fire.read()}, 201
            except Exception as e:
                return {"error": str(e)}, 500

    class _Read(Resource):
        def get(self, record_id=None):
            """Read forest fire record(s)"""
            try:
                # If ID is provided, return specific forest fire record
                if record_id:
                    forest_fire = ForestFire.query.get(record_id)
                    if forest_fire:
                        return forest_fire.read()
                    return {"error": "Forest fire record not found"}, 404

                # Otherwise return all forest fire records
                forest_fires = ForestFire.query.all()
                return jsonify([forest_fire.read() for forest_fire in forest_fires])
            except Exception as e:
                return {"error": str(e)}, 500

    class _Update(Resource):
        def put(self, record_id):
            """Update a forest fire record"""
            try:
                # Find forest fire record by ID
                forest_fire = ForestFire.query.get(record_id)
                if not forest_fire:
                    return {"error": "Forest fire record not found"}, 404

                # Get data from request body
                forest_data = request.get_json()

                # Update forest fire attributes
                for key in forest_data:
                    if hasattr(forest_fire, key):
                        setattr(forest_fire, key, forest_data[key])

                # Save changes
                db.session.commit()
                return {"message": "Forest fire record updated successfully", "forest_fire": forest_fire.read()}
            except Exception as e:
                db.session.rollback()
                return {"error": str(e)}, 500

    class _Delete(Resource):
        def delete(self, record_id):
            """Delete a forest fire record"""
            try:
                # Find forest fire record by ID
                forest_fire = ForestFire.query.get(record_id)
                if not forest_fire:
                    return {"error": "Forest fire record not found"}, 404

                # Delete forest fire record
                forest_fire.delete()
                return {"message": f"Forest fire record {record_id} deleted successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    class _CalculateIndices(Resource):
        def post(self):
            """Calculate fire indices based on weather data"""
            try:
                # Get weather data from request body
                weather_data = request.get_json()

                # Validate required fields
                required_fields = ['temp', 'RH', 'wind', 'rain']
                for field in required_fields:
                    if field not in weather_data:
                        return {"error": f"Missing required field: {field}"}, 400

                # Get model instance
                forestFireModel = ForestFireModel.get_instance()

                # Calculate indices
                ffmc = forestFireModel.estimate_FFMC(
                    weather_data['temp'],
                    weather_data['RH'],
                    weather_data['wind'],
                    weather_data['rain']
                )
                dmc = forestFireModel.estimate_DMC(
                    weather_data['temp'],
                    weather_data['RH'],
                    weather_data['rain']
                )
                dc = forestFireModel.estimate_DC(
                    weather_data['temp'],
                    weather_data['rain']
                )
                isi = forestFireModel.estimate_ISI(ffmc, weather_data['wind'])

                # Return calculated indices
                return {
                    "FFMC": float(ffmc),
                    "DMC": float(dmc),
                    "DC": float(dc),
                    "ISI": float(isi)
                }
            except Exception as e:
                return {"error": str(e)}, 500

    class _Restore(Resource):
        def post(self):
            """Restore forest fire records from a list of dictionaries"""
            try:
                # Get data from request body
                records_data = request.get_json()

                if not isinstance(records_data, list):
                    return {"error": "Expected a list of forest fire data"}, 400

                # Restore records
                restored_count = 0
                for record in records_data:
                    try:
                        forest_fire = ForestFire(
                            X=record.get('X'),
                            Y=record.get('Y'),
                            month=record.get('month'),
                            day=record.get('day'),
                            FFMC=record.get('FFMC'),
                            DMC=record.get('DMC'),
                            DC=record.get('DC'),
                            ISI=record.get('ISI'),
                            temp=record.get('temp'),
                            RH=record.get('RH'),
                            wind=record.get('wind'),
                            rain=record.get('rain'),
                            area=record.get('area', 0.0)
                        )
                        forest_fire.create()
                        restored_count += 1
                    except Exception as e:
                        continue

                return {"message": f"Restored {restored_count} forest fire records successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    # Register endpoints
    api.add_resource(_Predict, '/predict')
    api.add_resource(_Create, '/record')
    api.add_resource(_Read, '/records', '/record/<int:record_id>')
    api.add_resource(_Update, '/record/<int:record_id>')
    api.add_resource(_Delete, '/record/<int:record_id>')
    api.add_resource(_CalculateIndices, '/calculate-indices')
    api.add_resource(_Restore, '/restore')