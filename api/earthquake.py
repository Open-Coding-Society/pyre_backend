from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.earthquake import EarthquakeModel, Earthquake
from __init__ import app, db
from datetime import datetime, timedelta
import pandas as pd

earthquake_api = Blueprint('earthquake_api', __name__,
                   url_prefix='/api/earthquake')

api = Api(earthquake_api)

class EarthquakeAPI:
    """Main API class for earthquake-related endpoints"""
    
    class _Resource(Resource):
        """Base endpoint for earthquake data"""
        def get(self):
            try:
                # Read and process the CSV file directly
                df = pd.read_csv('earthquakes.csv')
                df['time'] = pd.to_datetime(df['time'])
                
                # Get earthquakes from the last 24 hours
                yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
                recent_quakes = df[df['time'] >= yesterday]
                
                # Calculate magnitude categories
                magnitude_counts = {
                    'Major (7.0+)': 0,
                    'Strong (5.0-6.9)': 0,
                    'Moderate (3.0-4.9)': 0,
                    'Minor (<3.0)': 0
                }
                
                for _, quake in recent_quakes.iterrows():
                    mag = float(quake['mag'])
                    if mag >= 7.0:
                        magnitude_counts['Major (7.0+)'] += 1
                    elif mag >= 5.0:
                        magnitude_counts['Strong (5.0-6.9)'] += 1
                    elif mag >= 3.0:
                        magnitude_counts['Moderate (3.0-4.9)'] += 1
                    else:
                        magnitude_counts['Minor (<3.0)'] += 1

                return {
                    'recent_earthquakes': recent_quakes.to_dict('records'),
                    'category_counts': magnitude_counts,
                    'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                return {'error': str(e)}, 500

    class _Predict(Resource):
        def post(self):
            """Predict earthquake magnitude based on location and time data"""
            # Get the earthquake data from the request
            earthquake_data = request.get_json()

            # Validate the earthquake data
            required_fields = ['latitude', 'longitude', 'depth']
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
            required_fields = ['time', 'latitude', 'longitude', 'depth', 'mag']
            for field in required_fields:
                if field not in earthquake_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Create new earthquake instance
            try:
                # Parse the time string to datetime
                time = datetime.fromisoformat(earthquake_data['time'].replace('Z', '+00:00'))
                
                earthquake = Earthquake(
                    time=time,
                    latitude=earthquake_data['latitude'],
                    longitude=earthquake_data['longitude'],
                    depth=earthquake_data['depth'],
                    mag=earthquake_data['mag'],
                    magType=earthquake_data.get('magType'),
                    place=earthquake_data.get('place'),
                    type=earthquake_data.get('type', 'earthquake')
                )
                earthquake.create()
                return {"message": "Earthquake record created successfully", "earthquake": earthquake.read()}, 201
            except Exception as e:
                return {"error": str(e)}, 500

    class _Read(Resource):
        def get(self, record_id=None):
            """Read earthquake record(s)"""
            try:
                # Read the CSV file
                df = pd.read_csv('earthquakes.csv')
                df['time'] = pd.to_datetime(df['time'])
                
                # If ID is provided, return specific earthquake record
                if record_id is not None:
                    if record_id < 0 or record_id >= len(df):
                        return {"error": "Earthquake record not found"}, 404
                    record = df.iloc[record_id]
                    return {
                        'id': record_id,
                        'time': record['time'].isoformat(),
                        'latitude': float(record['latitude']),
                        'longitude': float(record['longitude']),
                        'depth': float(record['depth']),
                        'mag': float(record['mag']),
                        'magType': record.get('magType', None),
                        'place': record.get('place', None),
                        'type': record.get('type', 'earthquake')
                    }

                # Otherwise return all earthquake records
                records = []
                for idx, row in df.iterrows():
                    records.append({
                        'id': idx,
                        'time': row['time'].isoformat(),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'depth': float(row['depth']),
                        'mag': float(row['mag']),
                        'magType': row.get('magType', None),
                        'place': row.get('place', None),
                        'type': row.get('type', 'earthquake')
                    })
                return jsonify(records)
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
                    if key == 'time':
                        # Parse the time string to datetime
                        time = datetime.fromisoformat(earthquake_data['time'].replace('Z', '+00:00'))
                        setattr(earthquake, key, time)
                    elif hasattr(earthquake, key):
                        setattr(earthquake, key, earthquake_data[key])

                # Save changes
                db.session.commit()
                return {"message": "Earthquake record updated successfully", "earthquake": earthquake.read()}
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

    class _RiskFactors(Resource):
        def post(self):
            """Calculate earthquake risk factors"""
            try:
                geological_data = request.get_json()
                
                required_fields = ['latitude', 'longitude', 'depth', 
                                 'previous_magnitude', 'distance_to_fault']
                for field in required_fields:
                    if field not in geological_data:
                        return {"error": f"Missing required field: {field}"}, 400

                earthquakeModel = EarthquakeModel.get_instance()
                
                seismic_intensity = earthquakeModel.estimate_seismic_intensity(
                    geological_data['previous_magnitude'],
                    geological_data['depth'],
                    geological_data['distance_to_fault']
                )
                ground_acceleration = earthquakeModel.estimate_ground_acceleration(
                    seismic_intensity,
                    geological_data['distance_to_fault']
                )
                liquefaction_potential = earthquakeModel.estimate_liquefaction_potential(
                    geological_data['latitude'],
                    geological_data['longitude']
                )

                return {
                    "seismic_intensity": float(seismic_intensity),
                    "ground_acceleration": float(ground_acceleration),
                    "liquefaction_potential": float(liquefaction_potential)
                }
            except Exception as e:
                return {'error': str(e)}, 500

    class _Search(Resource):
        def get(self):
            """Search for earthquakes based on criteria"""
            try:
                # Get search parameters from query string
                min_mag = request.args.get('min_mag', type=float)
                max_mag = request.args.get('max_mag', type=float)
                start_time = request.args.get('start_time')
                end_time = request.args.get('end_time')
                min_lat = request.args.get('min_lat', type=float)
                max_lat = request.args.get('max_lat', type=float)
                min_lon = request.args.get('min_lon', type=float)
                max_lon = request.args.get('max_lon', type=float)

                # Build the query
                query = Earthquake.query

                if min_mag is not None:
                    query = query.filter(Earthquake.mag >= min_mag)
                if max_mag is not None:
                    query = query.filter(Earthquake.mag <= max_mag)
                if start_time:
                    start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    query = query.filter(Earthquake.time >= start)
                if end_time:
                    end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    query = query.filter(Earthquake.time <= end)
                if min_lat is not None:
                    query = query.filter(Earthquake.latitude >= min_lat)
                if max_lat is not None:
                    query = query.filter(Earthquake.latitude <= max_lat)
                if min_lon is not None:
                    query = query.filter(Earthquake.longitude >= min_lon)
                if max_lon is not None:
                    query = query.filter(Earthquake.longitude <= max_lon)

                # Execute query and return results
                earthquakes = query.all()
                return jsonify([earthquake.read() for earthquake in earthquakes])
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
                        # Parse the time string to datetime
                        time = datetime.fromisoformat(record.get('time', '').replace('Z', '+00:00'))
                        
                        earthquake = Earthquake(
                            time=time,
                            latitude=record.get('latitude'),
                            longitude=record.get('longitude'),
                            depth=record.get('depth'),
                            mag=record.get('mag'),
                            magType=record.get('magType'),
                            place=record.get('place'),
                            type=record.get('type', 'earthquake')
                        )
                        earthquake.create()
                        restored_count += 1
                    except Exception as e:
                        continue

                return {"message": f"Restored {restored_count} earthquake records successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

# Register endpoints
api.add_resource(EarthquakeAPI._Resource, '/')
api.add_resource(EarthquakeAPI._Predict, '/predict')
api.add_resource(EarthquakeAPI._Create, '/record')
api.add_resource(EarthquakeAPI._Read, '/records', '/record/<int:record_id>')
api.add_resource(EarthquakeAPI._Update, '/record/<int:record_id>')
api.add_resource(EarthquakeAPI._Delete, '/record/<int:record_id>')
api.add_resource(EarthquakeAPI._RiskFactors, '/risk-factors')
api.add_resource(EarthquakeAPI._Search, '/search')
api.add_resource(EarthquakeAPI._Restore, '/restore')