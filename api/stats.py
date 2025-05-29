from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import requests

# Setup Blueprint
stats_api = Blueprint('stats_api', __name__, url_prefix='/api/stats')
api = Api(stats_api)

class FireStatsApi:
    class _HumanCaused(Resource):
        def get(self):
            try:
                url = "https://www.nifc.gov/fire-information/statistics/human-caused"
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                data_table = soup.find("table")
                rows = data_table.find_all("tr")

                stats = []
                for row in rows[1:]:
                    cols = row.find_all("td")
                    year_data = {
                        "year": cols[0].text.strip(),
                        "human_caused_fires": cols[1].text.strip(),
                        "acres_burned": cols[2].text.strip()
                    }
                    stats.append(year_data)

                return jsonify({"source": url, "data": stats})

            except Exception as e:
                return {"error": "Failed to fetch human-caused fire data", "message": str(e)}, 500

    class _TotalWildfires(Resource):
        def get(self):
            try:
                url = "https://www.nifc.gov/fire-information/statistics/wildfires"
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                data_table = soup.find("table")
                rows = data_table.find_all("tr")

                stats = []
                for row in rows[1:]:
                    cols = row.find_all("td")
                    year_data = {
                        "year": cols[0].text.strip(),
                        "total_fires": cols[1].text.strip(),
                        "total_acres": cols[2].text.strip()
                    }
                    stats.append(year_data)

                return jsonify({"source": url, "data": stats})

            except Exception as e:
                return {"error": "Failed to fetch wildfire stats", "message": str(e)}, 500

# Register endpoints
api.add_resource(FireStatsApi._HumanCaused, '/human_caused')
api.add_resource(FireStatsApi._TotalWildfires, '/total_wildfires')
