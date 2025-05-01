from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.usettings import Settings
from api.jwt_authorize import token_required
from dotenv import load_dotenv
import os
import requests

load_dotenv()

weather_api = Blueprint('weather_api', __name__, url_prefix='/api')
api = Api(weather_api)

class WeatherApi:
    class _PREDICT(Resource):
        pass
    class _WEATHER(Resource):
        def get(self):
            lat = request.args.get('lat')
            lon = request.args.get('lon')

            if not lat or not lon:
                return {"error": "Missing required parameters",
                        "message": "Please provide latitude (lat) and longitude (lon)"}, 400

            try:
                api_key = os.environ.get('WEATHER_API_KEY')
                if not api_key:
                    return {"error": "Weather API key not configured"}, 500

                weather_url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': api_key,
                    'units': 'metric'
                }

                response = requests.get(weather_url, params=params)
                if response.status_code == 200:
                    weather_data = response.json()

                    fire_relevant_data = {
                        "location": {
                            "lat": lat,
                            "lon": lon,
                            "name": weather_data.get('name', 'Unknown')
                        },
                        "weather": {
                            "temperature": weather_data['main']['temp'],
                            "humidity": weather_data['main']['humidity'],
                            "wind_speed": weather_data['wind']['speed'],
                            "wind_direction": weather_data['wind'].get('deg', 0),
                            "precipitation": weather_data.get('rain', {}).get('1h', 0),
                            "conditions": weather_data['weather'][0]['main'],
                            "description": weather_data['weather'][0]['description']
                        },
                        "timestamp": weather_data['dt']
                    }

                    return fire_relevant_data
                else:
                    return {"error": "Weather API error",
                            "message": response.text}, response.status_code

            except Exception as e:
                return {"error": "Server error", "message": str(e)}, 500

api.add_resource(WeatherApi._WEATHER, '/get_weather')