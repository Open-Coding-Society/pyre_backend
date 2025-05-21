from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.earthquake import EarthquakeModel, Earthquake
from __init__ import app, db
import pandas as pd

earthquake_api = Blueprint('earthquake_api', __name__,
                       url_prefix='/api/earthquake')

api = Api(earthquake_api)

class EarthquakeAPI:
    class _Predict(Resource):
        def post(self):
            """Predict earthquake magnitude based on input parameters"""
            # Get the earthquake data from the request
            earthquake_data = request.get_json()

            # Validate the earthquake data
            required_fields = ['latitude', 'longitude', 'depth', 'time_of_day', 
                             'previous_magnitude', 'distance_to_fault']
            for field in required_fields:
                if field not in earthquake_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Get the singleton instance of the EarthquakeModel
            earthquakeModel = EarthquakeModel.get_instance()
            # Predict the magnitude
            response = earthquakeModel.predict(earthquake_data)

            # Return the response as JSON
            return jsonify(response)

    class _Create(Resource):
        def post(self):
            """Create a new earthquake record"""
            # Get data from request body
            earthquake_data = request.get_json()

            # Validate the earthquake data
            required_fields = ['latitude', 'longitude', 'depth', 'time_of_day', 
                             'previous_magnitude', 'distance_to_fault', 
                             'plate_boundary_type', 'soil_type', 'magnitude']
            for field in required_fields:
                if field not in earthquake_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Create new earthquake instance
            try:
                earthquake = Earthquake(
                    latitude=earthquake_data['latitude'],
                    longitude=earthquake_data['longitude'],
                    depth=earthquake_data['depth'],
                    time_of_day=earthquake_data['time_of_day'],
                    previous_magnitude=earthquake_data['previous_magnitude'],
                    distance_to_fault=earthquake_data['distance_to_fault'],
                    plate_boundary_type=earthquake_data['plate_boundary_type'],
                    soil_type=earthquake_data['soil_type'],
                    magnitude=earthquake_data['magnitude']
                )
                earthquake.create()
                return {"message": "Earthquake record created successfully", 
                        "earthquake": earthquake.read()}, 201
            except Exception as e:
                return {"error": str(e)}, 500

    class _Read(Resource):
        def get(self, record_id=None):
            """Read earthquake record(s)"""
            try:
                # If ID is provided, return specific earthquake record
                if record_id:
                    earthquake = Earthquake.query.get(record_id)
                    if earthquake:
                        return earthquake.read()
                    return {"error": "Earthquake record not found"}, 404

                # Otherwise return all earthquake records
                earthquakes = Earthquake.query.all()
                return jsonify([earthquake.read() for earthquake in earthquakes])
            except Exception as e:
                return {"error": str(e)}, 500

    class _Update(Resource):
        def put(self, record_id):
            """Update an earthquake record"""
            try:
                # Find earthquake record by ID
                earthquake = Earthquake.query.get(record_id)
                if not earthquake:
                    return {"error": "Earthquake record not found"}, 404

                # Get data from request body
                earthquake_data = request.get_json()

                # Update earthquake attributes
                for key in earthquake_data:
                    if hasattr(earthquake, key):
                        setattr(earthquake, key, earthquake_data[key])

                # Save changes
                db.session.commit()
                return {"message": "Earthquake record updated successfully", 
                        "earthquake": earthquake.read()}
            except Exception as e:
                db.session.rollback()
                return {"error": str(e)}, 500

    class _Delete(Resource):
        def delete(self, record_id):
            """Delete an earthquake record"""
            try:
                # Find earthquake record by ID
                earthquake = Earthquake.query.get(record_id)
                if not earthquake:
                    return {"error": "Earthquake record not found"}, 404

                # Delete earthquake record
                earthquake.delete()
                return {"message": f"Earthquake record {record_id} deleted successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    class _CalculateRiskIndex(Resource):
        def post(self):
            """Calculate earthquake risk index based on input parameters"""
            try:
                # Get data from request body
                data = request.get_json()

                # Validate required fields
                required_fields = ['magnitude', 'depth', 'distance_to_fault', 'soil_type']
                for field in required_fields:
                    if field not in data:
                        return {"error": f"Missing required field: {field}"}, 400

                # Get model instance
                earthquakeModel = EarthquakeModel.get_instance()

                # Calculate risk index
                risk_index = earthquakeModel.calculate_risk_index(
                    data['magnitude'],
                    data['depth'],
                    data['distance_to_fault'],
                    data['soil_type']
                )

                return {
                    "risk_index": float(risk_index),
                    "risk_level": earthquakeModel.get_risk_level(risk_index)
                }
            except Exception as e:
                return {"error": str(e)}, 500

    class _Restore(Resource):
        def post(self):
            """Restore earthquake records from a list of dictionaries"""
            try:
                # Get data from request body
                records_data = request.get_json()

                if not isinstance(records_data, list):
                    return {"error": "Expected a list of earthquake data"}, 400

                # Restore records
                restored_count = 0
                for record in records_data:
                    try:
                        earthquake = Earthquake(
                            latitude=record.get('latitude'),
                            longitude=record.get('longitude'),
                            depth=record.get('depth'),
                            time_of_day=record.get('time_of_day'),
                            previous_magnitude=record.get('previous_magnitude'),
                            distance_to_fault=record.get('distance_to_fault'),
                            plate_boundary_type=record.get('plate_boundary_type'),
                            soil_type=record.get('soil_type'),
                            magnitude=record.get('magnitude')
                        )
                        earthquake.create()
                        restored_count += 1
                    except Exception:
                        continue

                return {"message": f"Restored {restored_count} earthquake records successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    # Register endpoints
    api.add_resource(_Predict, '/predict')
    api.add_resource(_Create, '/record')
    api.add_resource(_Read, '/records', '/record/<int:record_id>')
    api.add_resource(_Update, '/record/<int:record_id>')
    api.add_resource(_Delete, '/record/<int:record_id>')
    api.add_resource(_CalculateRiskIndex, '/calculate-risk-index')
    api.add_resource(_Restore, '/restore')