from flask import Blueprint, request, jsonify
from flask_restful import Api
from flask_restful import Resource
from __init__ import db
from model.usettings import Settings
from api.jwt_authorize import token_required
from dotenv import load_dotenv
import os
import requests

load_dotenv()

current_api = Blueprint('current_api', __name__, url_prefix='/api')
api = Api(current_api)
api_key = "51b8d0134b8e3ec9bcba5856741b34a1"

class CurrentApi(Resource):
    def get(self):
        city = request.args.get('City')

        if not city:
            return {"error": "Missing required parameters",
                    "message": "Please provide city name"}, 400

        try:
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': api_key,
            }

            response = requests.get(weather_url, params=params)
            if response.status_code == 200:
                weather_data = response.json()

                fire_relevant_data = {
                    # "location": {
                    #     "lat": lat,
                    #     "lon": lon,
                    #     "name": weather_data.get('name', 'Unknown')
                    # },
                    "weather": {
                        "temperature": weather_data['main']['temp'],
                        "humidity": weather_data['main']['humidity'],
                        "wind_speed": weather_data['wind']['speed'],
                        "description": weather_data['weather']['description']
                    }
                }

                return fire_relevant_data
            else:
                return {"error": "Weather API error",
                        "message": response.text}, response.status_code

        except Exception as e:
            return {"error": "Server error", "message": str(e)}, 500

api.add_resource(CurrentApi, '/current_api')