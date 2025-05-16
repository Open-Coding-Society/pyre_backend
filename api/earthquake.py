import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

# Load the earthquakes.csv file
earthquake_data_df = pd.read_csv('earthquakes.csv')

app = Flask(__name__)
api = Api(app)

class EarthquakeModel:
    _instance = None

    @staticmethod
    def get_instance():
        if EarthquakeModel._instance is None:
            EarthquakeModel._instance = EarthquakeModel()
        return EarthquakeModel._instance

    def predict(self, earthquake_data):
        # Example prediction logic using the CSV data
        # For simplicity, return a dummy prediction
        return {"predicted_magnitude": 5.0}

class Predict(Resource):
    def post(self):
        """Predict earthquake magnitude"""
        # Get the earthquake data from the request
        earthquake_data = request.get_json()

        # Validate the earthquake data
        required_fields = ['latitude', 'longitude', 'depth', 'time_of_day', 'previous_magnitude', 'distance_to_fault']
        for field in required_fields:
            if field not in earthquake_data:
                return {"error": f"Missing required field: {field}"}, 400

        # Example: Check if the data matches any record in the CSV
        matching_records = earthquake_data_df[
            (earthquake_data_df['latitude'] == earthquake_data['latitude']) &
            (earthquake_data_df['longitude'] == earthquake_data['longitude'])
        ]

        if not matching_records.empty:
            return {"message": "Matching record found in CSV", "data": matching_records.to_dict(orient='records')}, 200

        # Get the singleton instance of the EarthquakeModel
        earthquakeModel = EarthquakeModel.get_instance()
        # Predict the magnitude
        response = earthquakeModel.predict(earthquake_data)

        # Return the response as JSON
        return jsonify(response)

class Create(Resource):
    def post(self):
        """Create a new earthquake record"""
        # Get data from request body
        earthquake_data = request.get_json()

        # Validate the earthquake data
        required_fields = ['latitude', 'longitude', 'depth', 'time_of_day', 'previous_magnitude', 
                          'distance_to_fault', 'plate_boundary_type', 'soil_type', 'magnitude']
        for field in required_fields:
            if field not in earthquake_data:
                return {"error": f"Missing required field: {field}"}, 400

        # Append the new record to the CSV file
        new_record = pd.DataFrame([earthquake_data])
        new_record.to_csv('earthquakes.csv', mode='a', header=False, index=False)

        return {"message": "Earthquake record created successfully"}, 201

# Add resources to the API
api.add_resource(Predict, '/predict')
api.add_resource(Create, '/create')

if __name__ == '__main__':
    app.run(debug=True)