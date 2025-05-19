from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.earthquake import Earthquake
from __init__ import db
import json

# Define Blueprint
earthquake_api = Blueprint('earthquake_api', __name__, url_prefix='/api/earthquake')
api = Api(earthquake_api)

# Valid earthquake types to include
VALID_TYPES = {'earthquake', 'explosion', 'nuclear explosion'}

class EarthquakeAPI:
    class _Create(Resource):
        def post(self):
            data = request.get_json()
            required_fields = ['time', 'latitude', 'longitude', 'depth', 'mag', 'type']

            for field in required_fields:
                if field not in data:
                    return {"error": f"Missing field: {field}"}, 400

            if data['type'] not in VALID_TYPES:
                return {"error": "Invalid earthquake type"}, 400

            try:
                quake = Earthquake(
                    time=data['time'],
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    depth=data['depth'],
                    mag=data['mag'],
                    magType=data.get('magType'),
                    place=data.get('place'),
                    type=data['type']
                )
                quake.create()
                return {"message": "Earthquake record created", "record": quake.read()}, 201
            except Exception as e:
                return {"error": str(e)}, 500

    class _Read(Resource):
        def get(self, record_id=None):
            try:
                if record_id:
                    quake = Earthquake.query.get(record_id)
                    if quake:
                        return quake.read()
                    return {"error": "Earthquake record not found"}, 404

                records = Earthquake.query.all()
                return jsonify([q.read() for q in records])
            except Exception as e:
                return {"error": str(e)}, 500

    class _Update(Resource):
        def put(self, record_id):
            quake = Earthquake.query.get(record_id)
            if not quake:
                return {"error": "Earthquake record not found"}, 404

            data = request.get_json()
            try:
                for key, value in data.items():
                    if hasattr(quake, key):
                        setattr(quake, key, value)
                db.session.commit()
                return {"message": "Record updated", "record": quake.read()}
            except Exception as e:
                db.session.rollback()
                return {"error": str(e)}, 500

    class _Delete(Resource):
        def delete(self, record_id):
            quake = Earthquake.query.get(record_id)
            if not quake:
                return {"error": "Earthquake record not found"}, 404

            try:
                quake.delete()
                return {"message": f"Record {record_id} deleted"}
            except Exception as e:
                return {"error": str(e)}, 500

    class _RestoreFromFile(Resource):
        def post(self):
            try:
                with open('data/earthquake_records.json') as f:
                    records = json.load(f)

                restored = 0
                for rec in records:
                    if rec.get('type') in VALID_TYPES:
                        quake = Earthquake(
                            time=rec.get('time'),
                            latitude=rec.get('latitude'),
                            longitude=rec.get('longitude'),
                            depth=rec.get('depth'),
                            mag=rec.get('mag'),
                            magType=rec.get('magType'),
                            place=rec.get('place'),
                            type=rec.get('type')
                        )
                        try:
                            quake.create()
                            restored += 1
                        except Exception:
                            continue
                return {"message": f"Restored {restored} records from file"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    # Register endpoints
    api.add_resource(_Create, '/record')
    api.add_resource(_Read, '/records', '/record/<int:record_id>')
    api.add_resource(_Update, '/record/<int:record_id>')
    api.add_resource(_Delete, '/record/<int:record_id>')
    api.add_resource(_RestoreFromFile, '/restore')
