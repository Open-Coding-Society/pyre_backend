from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
from model.titanic import TitanicModel, Passenger
from __init__ import app, db

titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')

api = Api(titanic_api)

class TitanicAPI:
    class _Predict(Resource):
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending passenger data to the server to get a prediction fits the semantics of a POST request.
            
            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """     
            # Get the passenger data from the request
            passenger_data = request.get_json()

            # Validate the passenger data
            required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']
            for field in required_fields:
                if field not in passenger_data:
                    return {"error": f"Missing required field: {field}"}, 400

            # Get the singleton instance of the TitanicModel
            titanicModel = TitanicModel.get_instance()
            # Predict the survival probability of the passenger
            response = titanicModel.predict(passenger_data)

            # Return the response as JSON
            return jsonify(response)
    
    class _Create(Resource):
        def post(self):
            """Create a new passenger record"""
            # Get data from request body
            passenger_data = request.get_json()
            
            # Validate the passenger data
            required_fields = ['name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']
            for field in required_fields:
                if field not in passenger_data:
                    return {"error": f"Missing required field: {field}"}, 400
            
            # Create new passenger instance
            try:
                passenger = Passenger(
                    name=passenger_data['name'],
                    pclass=passenger_data['pclass'],
                    sex=passenger_data['sex'],
                    age=passenger_data['age'],
                    sibsp=passenger_data['sibsp'],
                    parch=passenger_data['parch'],
                    fare=passenger_data['fare'],
                    embarked=passenger_data['embarked'],
                    alone=passenger_data['alone']
                )
                passenger.create()
                return {"message": "Passenger created successfully", "passenger": passenger.read()}, 201
            except Exception as e:
                return {"error": str(e)}, 500
    
    class _Read(Resource):
        def get(self, passenger_id=None):
            """Read passenger record(s)"""
            try:
                # If ID is provided, return specific passenger
                if passenger_id:
                    passenger = Passenger.query.get(passenger_id)
                    if passenger:
                        return passenger.read()
                    return {"error": "Passenger not found"}, 404
                
                # Otherwise return all passengers
                passengers = Passenger.query.all()
                return jsonify([passenger.read() for passenger in passengers])
            except Exception as e:
                return {"error": str(e)}, 500
    
    class _Update(Resource):
        def put(self, passenger_id):
            """Update a passenger record"""
            try:
                # Find passenger by ID
                passenger = Passenger.query.get(passenger_id)
                if not passenger:
                    return {"error": "Passenger not found"}, 404
                
                # Get data from request body
                passenger_data = request.get_json()
                
                # Update passenger attributes
                for key in passenger_data:
                    if hasattr(passenger, key):
                        setattr(passenger, key, passenger_data[key])
                
                # Save changes
                db.session.commit()
                return {"message": "Passenger updated successfully", "passenger": passenger.read()}
            except Exception as e:
                db.session.rollback()
                return {"error": str(e)}, 500
    
    class _Delete(Resource):
        def delete(self, passenger_id):
            """Delete a passenger record"""
            try:
                # Find passenger by ID
                passenger = Passenger.query.get(passenger_id)
                if not passenger:
                    return {"error": "Passenger not found"}, 404
                
                # Delete passenger
                passenger.delete()
                return {"message": f"Passenger {passenger_id} deleted successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500
    
    class _Restore(Resource):
        def post(self):
            """Restore passengers from a list of dictionaries"""
            try:
                # Get data from request body
                passengers_data = request.get_json()
                
                if not isinstance(passengers_data, list):
                    return {"error": "Expected a list of passenger data"}, 400
                
                # Restore passengers
                restored = Passenger.restore(passengers_data)
                return {"message": f"Restored {len(restored)} passengers successfully"}, 200
            except Exception as e:
                return {"error": str(e)}, 500

    # Register endpoints
    api.add_resource(_Predict, '/predict')
    api.add_resource(_Create, '/passenger')
    api.add_resource(_Read, '/passengers', '/passenger/<int:passenger_id>')
    api.add_resource(_Update, '/passenger/<int:passenger_id>')
    api.add_resource(_Delete, '/passenger/<int:passenger_id>')
    api.add_resource(_Restore, '/restore')


