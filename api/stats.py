import requests
from flask import Blueprint, jsonify, request
from flask_restful import Api, Resource
from api.jwt_authorize import token_required

# Setup
stats_api = Blueprint('stats_api', __name__, url_prefix='/api')
api = Api(stats_api)

# Static WeatherAPI settings
STATS_API_URL = "https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/"
STATS_API_KEY = "a5c78a9b1ae831d370494dacf4428024"
DEFAULT_LOCATION = "world"
SOURCE = "VIIRS_SNPP_NRT"
DAY_RANGE = 1  # must be an integer

class WeatherAPI:
    class _Current(Resource):
        # Option 1: Keep authentication required
        @token_required()
        def get(self):
            """
            Get data for area of world
            """
            response = requests.get(
                f"{STATS_API_URL}/current.json",
                params={
                    "MAP_KEY": STATS_API_KEY,
                    "SOURCE": SOURCE,
                    "AREA_COORDINATES": DEFAULT_LOCATION,
                    "DAY_RANGE": DAY_RANGE  # now an integer
                }
            )

            print("Status code:", response.status_code)
            print("Response text:", response.text)

            if response.status_code != 200:
                return {"message": "Failed to fetch fire data."}, response.status_code

            try:
                data = response.json()
            except Exception as e:
                return {
                    "message": "Invalid JSON from API.",
                    "error": str(e),
                    "raw_response": response.text
                }, 502

            data = response.json()
            current = data.get("current", {})

            # Extract key values
            result = {
                "bright_ti4": current.get("bright_ti4"),
                "temperature_f": current.get("temp_f"),
                "confidence": current.get("confidence"),
                "frp": current.get("frp")
            }

            return jsonify(result)
        
    # Register endpoints
    api.add_resource(_Current, '/stats/current')