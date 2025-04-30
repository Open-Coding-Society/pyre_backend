import requests
from flask import Blueprint, jsonify, request
from flask_restful import Api, Resource
from api.jwt_authorize import token_required

# Setup
weather_api = Blueprint('weather_api', __name__, url_prefix='/api/weather')
api = Api(weather_api)

# Static WeatherAPI settings
WEATHER_API_URL = "http://api.weatherapi.com/v1"
WEATHER_API_KEY = "aa2ef74d67c7438898715913253004"
DEFAULT_LOCATION = "San Diego, California"


class WeatherAPI:
    class _Current(Resource):
        # Option 1: Keep authentication required
        @token_required()
        def get(self):
            """
            Get current weather for San Diego, CA
            Usage: GET /api/weather/current
            """
            response = requests.get(
                f"{WEATHER_API_URL}/current.json",
                params={"key": WEATHER_API_KEY, "q": DEFAULT_LOCATION}
            )

            if response.status_code != 200:
                return {"message": "Failed to fetch weather data."}, response.status_code

            data = response.json()
            current = data.get("current", {})
            location = data.get("location", {})

            # Extract key climate values
            result = {
                "location": f"{location.get('name')}, {location.get('region')}",
                "temperature_c": current.get("temp_c"),
                "temperature_f": current.get("temp_f"),
                "condition": current.get("condition", {}).get("text"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "feelslike_c": current.get("feelslike_c"),
                "feelslike_f": current.get("feelslike_f"),
                "last_updated": current.get("last_updated")
            }

            return jsonify(result)
            
    class _AtCoordinates(Resource):
        # Option 1: Keep authentication required
        @token_required()
        def get(self):
            """
            Get weather at specific coordinates
            Usage: GET /api/weather/at?lat=32.7157&lng=-117.1611
            """
            # Get coordinates from request
            lat = request.args.get('lat')
            lng = request.args.get('lng')
            
            if not lat or not lng:
                return {"message": "Missing lat or lng parameters"}, 400
                
            # Create coordinate string for WeatherAPI
            coordinates = f"{lat},{lng}"
            
            response = requests.get(
                f"{WEATHER_API_URL}/current.json",
                params={"key": WEATHER_API_KEY, "q": coordinates}
            )
            
            if response.status_code != 200:
                return {"message": "Failed to fetch weather data."}, response.status_code
                
            data = response.json()
            current = data.get("current", {})
            location = data.get("location", {})
            
            # Extract key climate values
            result = {
                "location": f"{location.get('name')}, {location.get('region')}",
                "temperature_c": current.get("temp_c"),
                "temperature_f": current.get("temp_f"),
                "condition": current.get("condition", {}).get("text"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "feelslike_c": current.get("feelslike_c"),
                "feelslike_f": current.get("feelslike_f"),
                "last_updated": current.get("last_updated")
            }
            
            return jsonify(result)

    # Option 2: Create public endpoints (non-authenticated) for the weather dashboard
    class _PublicCurrent(Resource):
        # No authentication required
        def get(self):
            """
            Public endpoint: Get current weather for San Diego, CA
            Usage: GET /api/weather/public/current
            """
            response = requests.get(
                f"{WEATHER_API_URL}/current.json",
                params={"key": WEATHER_API_KEY, "q": DEFAULT_LOCATION}
            )

            if response.status_code != 200:
                return {"message": "Failed to fetch weather data."}, response.status_code

            data = response.json()
            current = data.get("current", {})
            location = data.get("location", {})

            # Extract key climate values
            result = {
                "location": f"{location.get('name')}, {location.get('region')}",
                "temperature_c": current.get("temp_c"),
                "temperature_f": current.get("temp_f"),
                "condition": current.get("condition", {}).get("text"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "feelslike_c": current.get("feelslike_c"),
                "feelslike_f": current.get("feelslike_f"),
                "last_updated": current.get("last_updated")
            }

            return jsonify(result)
            
    class _PublicAtCoordinates(Resource):
        # No authentication required
        def get(self):
            """
            Public endpoint: Get weather at specific coordinates
            Usage: GET /api/weather/public/at?lat=32.7157&lng=-117.1611
            """
            # Get coordinates from request
            lat = request.args.get('lat')
            lng = request.args.get('lng')
            
            if not lat or not lng:
                return {"message": "Missing lat or lng parameters"}, 400
                
            # Create coordinate string for WeatherAPI
            coordinates = f"{lat},{lng}"
            
            response = requests.get(
                f"{WEATHER_API_URL}/current.json",
                params={"key": WEATHER_API_KEY, "q": coordinates}
            )
            
            if response.status_code != 200:
                return {"message": "Failed to fetch weather data."}, response.status_code
                
            data = response.json()
            current = data.get("current", {})
            location = data.get("location", {})
            
            # Extract key climate values
            result = {
                "location": f"{location.get('name')}, {location.get('region')}",
                "temperature_c": current.get("temp_c"),
                "temperature_f": current.get("temp_f"),
                "condition": current.get("condition", {}).get("text"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "feelslike_c": current.get("feelslike_c"),
                "feelslike_f": current.get("feelslike_f"),
                "last_updated": current.get("last_updated")
            }
            
            return jsonify(result)

    # Register endpoints
    api.add_resource(_Current, '/current')
    api.add_resource(_AtCoordinates, '/at')
    
    # Register public endpoints (Option 2)
    api.add_resource(_PublicCurrent, '/public/current')
    api.add_resource(_PublicAtCoordinates, '/public/at')