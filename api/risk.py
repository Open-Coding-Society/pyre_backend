from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from datetime import datetime, timedelta
import math
from typing import Dict, Any, Tuple, Optional

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
ZIPCODE_API_KEY = os.getenv('ZIPCODE_API_KEY', '839839a0-3f38-11f0-a681-9f2ad9a37e25')
NASA_EARTHDATA_TOKEN = os.getenv('NASA_EARTHDATA_TOKEN', 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImFhcnVzaGdvd2RhIiwiZXhwIjoxNzU0MDAxNTE0LCJpYXQiOjE3NDg4MTc1MTQsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.l5SK4D-cenwSJPFy9wIpxQBiRNSY5RgDO_8POyw1Xz3Ne6NgBuMOXQ9r2v2YfhtTtcMS9sTcs7pVnXG3ogg8PUqv7gPzqOzHsqeNjdPG9uyIIpK3VwhM-OoTOIOD8PyGjk58ZHmcpwZWvusa7VMA0RyEioPGsQhUN8M1zRuF-P4WPolmGBpSAMHLLKGUqoN57CkYQZBtgu3kxjWYHc85XJ4b1-6FZmZms0WrZl3srMktvdgKQQh1QDa3L4FamjZJlkR9KiFRts4p8qxhgvSuAhFt-RarvDjpfSoKl7JTWd6BLTEr3e9ehUjEiRQFMfaaLS07UzgeCDKbe6wenEi1dw	')

class RiskPredictor:
    """Main class for calculating wildfire and earthquake risks"""
    
    def __init__(self):
        self.zip_api_base = "https://api.zippopotam.us/us"
        self.usgs_earthquake_api = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self.nasa_firms_api = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        
    def get_coordinates_from_zip(self, zip_code: str) -> Optional[Tuple[float, float]]:
        """Convert ZIP code to latitude/longitude coordinates"""
        try:
            response = requests.get(f"{self.zip_api_base}/{zip_code}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                lat = float(data['places'][0]['latitude'])
                lon = float(data['places'][0]['longitude'])
                return lat, lon
            return None
        except Exception as e:
            print(f"Error getting coordinates for ZIP {zip_code}: {e}")
            return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def get_wildfire_risk(self, zip_code: str) -> Dict[str, Any]:
        """Calculate wildfire risk for given ZIP code"""
        coords = self.get_coordinates_from_zip(zip_code)
        if not coords:
            return {"error": "Invalid ZIP code"}
        
        lat, lon = coords
        
        # Get recent fire data from USGS (using earthquake API as proxy for demonstration)
        # In production, you'd use NASA FIRMS or other fire APIs
        recent_fires = self._get_recent_fire_activity(lat, lon)
        
        # Calculate vegetation and weather risk factors
        vegetation_risk = self._calculate_vegetation_risk(lat, lon)
        weather_risk = self._calculate_weather_risk(lat, lon)
        
        # Calculate overall risk score (0-100)
        risk_score = self._calculate_wildfire_score(recent_fires, vegetation_risk, weather_risk)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "zip_code": zip_code,
            "coordinates": {"latitude": lat, "longitude": lon},
            "risk_level": risk_level,
            "risk_score": risk_score,
            "recent_fires": recent_fires,
            "vegetation_risk": vegetation_risk,
            "weather_risk": weather_risk,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_earthquake_risk(self, zip_code: str) -> Dict[str, Any]:
        """Calculate earthquake risk for given ZIP code"""
        coords = self.get_coordinates_from_zip(zip_code)
        if not coords:
            return {"error": "Invalid ZIP code"}
        
        lat, lon = coords
        
        # Get recent earthquake data from USGS
        recent_earthquakes, max_magnitude = self._get_recent_earthquakes(lat, lon)
        
        # Calculate fault proximity (simplified)
        fault_distance = self._calculate_fault_proximity(lat, lon)
        
        # Calculate overall risk score
        risk_score = self._calculate_earthquake_score(recent_earthquakes, max_magnitude, fault_distance)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "zip_code": zip_code,
            "coordinates": {"latitude": lat, "longitude": lon},
            "risk_level": risk_level,
            "risk_score": risk_score,
            "recent_earthquakes": recent_earthquakes,
            "max_magnitude": max_magnitude,
            "fault_distance": f"{fault_distance:.1f} km",
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_recent_fire_activity(self, lat: float, lon: float) -> int:
        """Get recent fire activity count (simplified implementation)"""
        # This is a simplified implementation
        # In production, you'd query NASA FIRMS or similar APIs
        
        # Simulate fire activity based on geographic factors
        # Higher activity in western US, dry regions
        if -125 <= lon <= -100 and 30 <= lat <= 50:  # Western US
            base_activity = 15
        elif -100 <= lon <= -80 and 25 <= lat <= 45:  # Central US
            base_activity = 5
        else:  # Eastern US
            base_activity = 2
        
        # Add some randomness and seasonal factors
        import random
        seasonal_factor = 1.5 if datetime.now().month in [6, 7, 8, 9] else 0.8
        return max(0, int(base_activity * seasonal_factor + random.randint(-3, 5)))
    
    def _calculate_vegetation_risk(self, lat: float, lon: float) -> str:
        """Calculate vegetation-based fire risk"""
        # Simplified vegetation risk based on geography
        if -125 <= lon <= -100 and 30 <= lat <= 50:  # Western US - high vegetation risk
            return "High"
        elif -100 <= lon <= -80:  # Central US - medium risk
            return "Medium"
        else:  # Eastern US - generally lower risk
            return "Low"
    
    def _calculate_weather_risk(self, lat: float, lon: float) -> str:
        """Calculate weather-based fire risk"""
        # Simplified weather risk (would use real weather APIs in production)
        current_month = datetime.now().month
        
        if -125 <= lon <= -100:  # Western US
            if current_month in [6, 7, 8, 9]:  # Fire season
                return "High"
            elif current_month in [4, 5, 10]:
                return "Medium"
            else:
                return "Low"
        else:
            return "Low"
    
    def _calculate_wildfire_score(self, recent_fires: int, vegetation_risk: str, weather_risk: str) -> int:
        """Calculate overall wildfire risk score (0-100)"""
        score = 0
        
        # Recent fire activity (0-40 points)
        score += min(40, recent_fires * 2)
        
        # Vegetation risk (0-30 points)
        vegetation_scores = {"Low": 5, "Medium": 15, "High": 30}
        score += vegetation_scores[vegetation_risk]
        
        # Weather risk (0-30 points)
        weather_scores = {"Low": 5, "Medium": 15, "High": 30}
        score += weather_scores[weather_risk]
        
        return min(100, score)
    
    def _get_recent_earthquakes(self, lat: float, lon: float) -> Tuple[int, float]:
        """Get recent earthquake data from USGS API"""
        try:
            # Define search area (1 degree radius)
            min_lat = lat - 1
            max_lat = lat + 1
            min_lon = lon - 1
            max_lon = lon + 1
            
            # Get earthquakes from the past year
            end_time = datetime.now()
            start_time = end_time - timedelta(days=365)
            
            params = {
                'format': 'geojson',
                'starttime': start_time.strftime('%Y-%m-%d'),
                'endtime': end_time.strftime('%Y-%m-%d'),
                'minlatitude': min_lat,
                'maxlatitude': max_lat,
                'minlongitude': min_lon,
                'maxlongitude': max_lon,
                'minmagnitude': 1.0
            }
            
            response = requests.get(self.usgs_earthquake_api, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                earthquakes = data['features']
                
                count = len(earthquakes)
                max_mag = 0.0
                
                if earthquakes:
                    magnitudes = [eq['properties']['mag'] for eq in earthquakes if eq['properties']['mag']]
                    max_mag = max(magnitudes) if magnitudes else 0.0
                
                return count, max_mag
            else:
                print(f"USGS API error: {response.status_code}")
                return 0, 0.0
                
        except Exception as e:
            print(f"Error fetching earthquake data: {e}")
            return 0, 0.0
    
    def _calculate_fault_proximity(self, lat: float, lon: float) -> float:
        """Calculate approximate distance to nearest major fault (simplified)"""
        # Major fault locations (simplified - San Andreas, New Madrid, etc.)
        major_faults = [
            (37.0, -122.0),  # San Andreas (Northern CA)
            (34.0, -118.0),  # San Andreas (Southern CA)
            (36.6, -89.6),   # New Madrid (Missouri)
            (47.0, -122.0),  # Seattle Fault
            (40.7, -111.9),  # Wasatch Fault (Utah)
        ]
        
        min_distance = float('inf')
        for fault_lat, fault_lon in major_faults:
            distance = self.calculate_distance(lat, lon, fault_lat, fault_lon)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_earthquake_score(self, recent_count: int, max_magnitude: float, fault_distance: float) -> int:
        """Calculate overall earthquake risk score (0-100)"""
        score = 0
        
        # Recent earthquake activity (0-30 points)
        score += min(30, recent_count * 2)
        
        # Maximum magnitude (0-40 points)
        if max_magnitude >= 6.0:
            score += 40
        elif max_magnitude >= 4.0:
            score += 25
        elif max_magnitude >= 2.0:
            score += 10
        
        # Fault proximity (0-30 points)
        if fault_distance < 50:
            score += 30
        elif fault_distance < 100:
            score += 20
        elif fault_distance < 200:
            score += 10
        else:
            score += 5
        
        return min(100, score)

# Initialize the risk predictor
risk_predictor = RiskPredictor()

@app.route('/api/risk/wildfire', methods=['GET'])
def get_wildfire_risk():
    """API endpoint for wildfire risk assessment"""
    zip_code = request.args.get('zip')
    
    if not zip_code:
        return jsonify({"error": "ZIP code parameter is required"}), 400
    
    if not zip_code.isdigit() or len(zip_code) != 5:
        return jsonify({"error": "Invalid ZIP code format"}), 400
    
    try:
        result = risk_predictor.get_wildfire_risk(zip_code)
        
        if "error" in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/risk/earthquake', methods=['GET'])
def get_earthquake_risk():
    """API endpoint for earthquake risk assessment"""
    zip_code = request.args.get('zip')
    
    if not zip_code:
        return jsonify({"error": "ZIP code parameter is required"}), 400
    
    if not zip_code.isdigit() or len(zip_code) != 5:
        return jsonify({"error": "Invalid ZIP code format"}), 400
    
    try:
        result = risk_predictor.get_earthquake_risk(zip_code)
        
        if "error" in result:
            return jsonify(result), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🔥 Risk Predictor API Server Starting...")
    print("Available endpoints:")
    print("  GET /api/risk/wildfire?zip=<zipcode>")
    print("  GET /api/risk/earthquake?zip=<zipcode>") 
    print("  GET /api/health")
    print("\nExample usage:")
    print("  curl http://localhost:8505/api/risk/wildfire?zip=92127")
    print("  curl http://localhost:8505/api/risk/earthquake?zip=92127")
    print("\nStarting server on http://localhost:8505...")
    
    app.run(debug=True, host='0.0.0.0', port=8505)