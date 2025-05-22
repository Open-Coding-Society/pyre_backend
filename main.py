# imports from flask
import json
import os
from urllib.parse import urljoin, urlparse
from flask import abort, redirect, render_template, request, send_from_directory, url_for, jsonify  # import render_template from "public" flask libraries
from flask_login import current_user, login_user, logout_user
from flask.cli import AppGroup
from flask_login import current_user, login_required
from flask import current_app
from werkzeug.security import generate_password_hash
import shutil
from functools import wraps
from twilio.rest import Client
from dotenv import load_dotenv
import time
from email.header import decode_header
import os
import base64
import threading
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

# import "objects" from "this" project
from __init__ import app, db, login_manager  # Key Flask objects 
# API endpoints
from api.user import user_api 
from api.pfp import pfp_api
from api.post import post_api
from api.usettings import settings_api
from api.user_met import user_met_api
from api.post_met import post_met_api
from api.titanic import titanic_api  # Import the titanic API
from api.weather import weather_api
from api.email import email_api
from api.fire import forest_fire_api
from api.stats import stats_api
from api.historical_fire import historical_fire_api
from api.help import help_api
from api.earthquake import earthquake_api  # Import the earthquake API

# database Initialization functions
from model.user import User, initUsers
from model.section import Section, initSections
from model.channel import Channel, initChannels
from model.post import Post, initPosts
from model.teaminfo import TeamMember, initTeamMembers
from model.usettings import Settings  # Import the Settings model
from model.titanic import TitanicModel  # Import the TitanicModel class
from model.titanic import Passenger, initPassengers
from model.email import Email, initEmail
from model.fire import ForestFireModel, ForestFire
from model.earthquake import Earthquake, initEarthquakes  # Add this import
from model.historical_fire import FireDataAnalysisAdvancedRegressionModel
from model.help import initHelpSystem
# server only Views

# register URIs for api endpoints
app.register_blueprint(user_api)
app.register_blueprint(pfp_api)
app.register_blueprint(post_api)
app.register_blueprint(user_met_api)
app.register_blueprint(post_met_api)
app.register_blueprint(titanic_api)
app.register_blueprint(weather_api)
app.register_blueprint(email_api)
app.register_blueprint(forest_fire_api)
app.register_blueprint(stats_api)
app.register_blueprint(earthquake_api)  # Register the earthquake API
app.register_blueprint(historical_fire_api)
app.register_blueprint(help_api)

load_dotenv()

# Tell Flask-Login the view function name of your login route
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login', next=request.path))

# register URIs for server pages
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# Helper function to check if the URL is safe for redirects
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'Admin':
            return redirect(url_for('unauthorized'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    next_page = request.args.get('next', '') or request.form.get('next', '')
    if request.method == 'POST':
        user = User.query.filter_by(_uid=request.form['username']).first()
        if user and user.is_password(request.form['password']):
            login_user(user)
            if not is_safe_url(next_page):
                return abort(400)
            if user.role == 'Admin':
                return redirect(next_page or url_for('index'))
            else:
                return redirect(next_page or url_for('user_index'))
        else:
            error = 'Invalid username or password.'
    return render_template("login.html", error=error, next=next_page)
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route('/')  # connects default URL to index() function
def index():
    if current_user.is_authenticated and current_user.role == 'Admin':
        return render_template("index.html")
    elif current_user.is_authenticated:
        return render_template("user_index.html")
    return render_template("login.html")

@app.route('/unauthorized')
def unauthorized():
    return render_template('unauthorized.html'), 401

@app.route('/user_index')
@login_required
def user_index():
    return render_template("user_index.html")

@app.route('/users/table')
@login_required
def utable():
    users = User.query.all()
    return render_template("utable.html", user_data=users)

@app.route('/users/table2')
@login_required
def u2table():
    users = User.query.all()
    return render_template("u2table.html", user_data=users)

@app.route('/postdata')
@admin_required
@login_required
def postData():
    users = User.query.all()
    return render_template("postData.html", user_data=users)

@app.route('/users/settings')
@admin_required
@login_required
def usettings():
    users = User.query.all()
    return render_template("usettings.html", user_data=users)

@app.route('/users/reports')
@admin_required
@login_required
def ureports():
    users = User.query.all()
    return render_template("ureports.html", user_data=users)

@app.route('/general-settings', methods=['GET', 'POST'])
@login_required
@admin_required
def general_settings():
    settings = Settings.query.first()
    if request.method == 'POST':
        settings.description = request.form['description']
        settings.contact_email = request.form['contact_email']
        settings.contact_phone = request.form['contact_phone']
        db.session.commit()
        return redirect(url_for('general_settings'))
    return render_template('ugeneralsettings.html', settings=settings)

# Helper function to extract uploads for a user (ie PFP image)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
 
@app.route('/users/delete/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get(user_id)
    if user:
        user.delete()
        return jsonify({'message': 'User deleted successfully'}), 200
    return jsonify({'error': 'User not found'}), 404

@app.route('/users/reset_password/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reset_password(user_id):
    if current_user.role != 'Admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Set the new password
    if user.update({"password": app.config['DEFAULT_PASSWORD']}):
        return jsonify({'message': 'Password reset successfully'}), 200
    return jsonify({'error': 'Password reset failed'}), 500

##################### TWILIO CALLS #####################

account_sid = os.environ.get('ACCOUNT_SID')
auth_token = os.environ.get('AUTH_TOKEN')
twilio_number = os.environ.get('TWILIO_NUMBER')

client = Client(account_sid, auth_token)

@app.route('/make-call', methods=['POST'])
def make_call():
    data = request.json
    to_number = data.get('to_number')

    if not to_number:
        return jsonify({"error": "No phone number provided"}), 400

    try:
        call = client.calls.create(
            to=to_number,
            from_=twilio_number,
            # This URL will be called when the call connects
            # It should return TwiML instructions for the call
            url="https://handler.twilio.com/twiml/EH031d8c1ce0bc7bf9404865b155053aab"
        )
        return jsonify({
            "success": True,
            "call_sid": call.sid,
            "status": call.status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

##################### FIRE INCIDENTS MONITORING #####################

# Global variables
URL = "https://seshat.datasd.org/fire_ems_incidents/fd_incidents_2025_datasd.csv"
FILENAME = "fire_incidents.csv"
UPDATE_INTERVAL = 43200  # 12 hours in seconds
last_download_time = None
data_lock = threading.Lock()  # Lock for thread safety

# Cached data
cached_problem_counts = {}
cached_category_counts = {}
cached_incidents_list = []

def download_csv():
    global last_download_time
    
    try:
        logger.info(f"Downloading fire incidents data from {URL}")
        response = requests.get(URL, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(FILENAME, 'wb') as f:
            f.write(response.content)
        
        last_download_time = datetime.now()
        logger.info(f"Download successful. File saved to {FILENAME}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading CSV: {e}")
        return False

def parse_fire_data():
    global cached_problem_counts, cached_category_counts, cached_incidents_list
    
    try:
        logger.info(f"Parsing fire data from {FILENAME}")
        df = pd.read_csv(FILENAME)
        df['date_response'] = pd.to_datetime(df['date_response'], errors='coerce')
        today = pd.Timestamp.now().normalize()
        today_incidents = df[df['date_response'].dt.normalize() == today]
        
        with data_lock:
            cached_problem_counts = today_incidents['problem'].value_counts().to_dict()
            cached_category_counts = today_incidents['call_category'].value_counts().to_dict()
            cached_incidents_list = today_incidents.to_dict(orient="records")
        
        logger.info(f"Parsed data successfully. Found {len(today_incidents)} incidents for today.")
        return True
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        return False

def update_data():
    if download_csv():
        parse_fire_data()

def fire_monitor_thread():
    logger.info("Starting fire monitoring thread")
    
    # Initial data download and parse
    update_data()
    
    while True:
        try:
            logger.info(f"Next data update in {UPDATE_INTERVAL/3600} hours")
            time.sleep(UPDATE_INTERVAL)
            logger.info("Checking for new fire incidents...")
            update_data()
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
            # Sleep briefly before trying again to avoid tight loop on error
            time.sleep(60)

@app.route('/fire-resource', methods=['GET'])
def fire_resource():
    with data_lock:
        response_data = {
            "problem_counts": cached_problem_counts,
            "category_counts": cached_category_counts,
            "today_incidents": cached_incidents_list,
            "last_update": last_download_time.isoformat() if last_download_time else None
        }
    
    return jsonify(response_data)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "last_update": last_download_time.isoformat() if last_download_time else None
    })

@app.route('/refresh', methods=['GET'])
def refresh_data():
    threading.Thread(target=update_data).start()
    return jsonify({"status": "refresh started"})

##################### EARTHQUAKE MONITORING #####################

# Global variables for earthquake monitoring
EARTHQUAKE_UPDATE_INTERVAL = 3600  # 1 hour in seconds
last_earthquake_update = None
earthquake_data_lock = threading.Lock()

# Cached earthquake data
cached_magnitude_counts = {}
cached_region_counts = {}
cached_recent_earthquakes = []

def parse_earthquake_data():
    global cached_magnitude_counts, cached_region_counts, cached_recent_earthquakes
    
    try:
        logger.info("Parsing earthquake data from earthquakes.csv")
        df = pd.read_csv('earthquakes.csv')
        df['time'] = pd.to_datetime(df['time'])
        
        # Get earthquakes from the last 24 hours
        now = pd.Timestamp.now()
        recent_time = now - timedelta(days=1)
        recent_quakes = df[df['time'] >= recent_time]
        
        with earthquake_data_lock:
            # Count earthquakes by magnitude ranges
            magnitude_ranges = pd.cut(recent_quakes['mag'], 
                                    bins=[-float('inf'), 2, 3, 4, 5, float('inf')],
                                    labels=['<2', '2-3', '3-4', '4-5', '>5'])
            cached_magnitude_counts = magnitude_ranges.value_counts().to_dict()
            
            # Count earthquakes by region (using 'place' field)
            cached_region_counts = recent_quakes['place'].value_counts().head(10).to_dict()
            
            # Store recent earthquakes
            cached_recent_earthquakes = recent_quakes.sort_values('time', ascending=False).head(50).to_dict('records')
        
        logger.info(f"Parsed earthquake data successfully. Found {len(recent_quakes)} earthquakes in the last 24 hours.")
        return True
    except Exception as e:
        logger.error(f"Error parsing earthquake data: {e}")
        return False

def update_earthquake_data():
    global last_earthquake_update
    parse_earthquake_data()
    last_earthquake_update = datetime.now()

def earthquake_monitor_thread():
    logger.info("Starting earthquake monitoring thread")
    
    # Initial data parse
    update_earthquake_data()
    
    while True:
        try:
            logger.info(f"Next earthquake data update in {EARTHQUAKE_UPDATE_INTERVAL/3600} hours")
            time.sleep(EARTHQUAKE_UPDATE_INTERVAL)
            logger.info("Checking for new earthquake data...")
            update_earthquake_data()
        except Exception as e:
            logger.error(f"Error in earthquake monitoring thread: {e}")
            time.sleep(60)  # Sleep briefly before trying again

@app.route('/earthquake-data', methods=['GET'])
def earthquake_data():
    with earthquake_data_lock:
        response_data = {
            "magnitude_counts": cached_magnitude_counts,
            "region_counts": cached_region_counts,
            "recent_earthquakes": cached_recent_earthquakes,
            "last_update": last_earthquake_update.isoformat() if last_earthquake_update else None
        }
    
    return jsonify(response_data)

##################### HISTORICAL FIRE DATA #####################

data_by_month = {}

def preprocess_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['acq_date'])
    df['year'] = df['acq_date'].dt.year
    df['month'] = df['acq_date'].dt.month
    
    grouped = df.groupby(['year', 'month'])
    for (year, month), group in grouped:
        key = f"{year}-{month:02d}"
        data_by_month[key] = group.to_dict(orient='records') 

# sample request: http://127.0.0.1:8505/get-historical-data?year=2015&month=01

@app.route("/get-historical-data", methods=["GET"])
def get_historical_data():
    year = request.args.get('year')
    month = request.args.get('month')
    key = f"{year}-{int(month):02d}"
    return jsonify(data_by_month.get(key, []))

##################### EARTHQUAKE HISTORICAL DATA #####################

earthquake_data_by_month = {}

def preprocess_earthquake_data(filepath):
    """Preprocess earthquake data and group by year-month"""
    try:
        logger.info(f"Preprocessing earthquake data from {filepath}")
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # The CSV columns based on your sample:
        # time, latitude, longitude, depth, mag, magType, nst, gap, dmin, rms, net, id, updated, place, type, horizontalError, depthError, magError, magNst, status, locationSource, magSource
        
        # Parse the time column
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['time'])
        
        # Extract year and month
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
        # Clean and validate data
        df = df.dropna(subset=['latitude', 'longitude'])  # Remove rows without coordinates
        df['mag'] = pd.to_numeric(df['mag'], errors='coerce')  # Ensure magnitude is numeric
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')  # Ensure depth is numeric
        
        # Group by year and month
        grouped = df.groupby(['year', 'month'])
        for (year, month), group in grouped:
            key = f"{year}-{month:02d}"
            earthquake_data_by_month[key] = group.to_dict(orient='records')
        
        logger.info(f"Earthquake data preprocessing complete. Found data for {len(earthquake_data_by_month)} month periods.")
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing earthquake data: {e}")
        return False

@app.route("/get-historical-earthquake-data", methods=["GET"])
def get_historical_earthquake_data():
    """Get historical earthquake data for a specific year and month"""
    year = request.args.get('year')
    month = request.args.get('month')
    
    if not year or not month:
        return jsonify({"error": "Year and month parameters are required"}), 400
    
    try:
        key = f"{year}-{int(month):02d}"
        data = earthquake_data_by_month.get(key, [])
        
        logger.info(f"Returning {len(data)} earthquake records for {key}")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting earthquake data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/earthquake-stats", methods=["GET"])
def earthquake_stats():
    """Get general statistics about available earthquake data"""
    try:
        total_months = len(earthquake_data_by_month)
        total_earthquakes = sum(len(data) for data in earthquake_data_by_month.values())
        
        available_periods = list(earthquake_data_by_month.keys())
        available_periods.sort()
        
        return jsonify({
            "total_months": total_months,
            "total_earthquakes": total_earthquakes,
            "available_periods": available_periods,
            "date_range": {
                "start": available_periods[0] if available_periods else None,
                "end": available_periods[-1] if available_periods else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting earthquake stats: {e}")
        return jsonify({"error": str(e)}), 500

# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to run the data generation functions
@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initSections()
    initPosts()
    initTeamMembers()
    initPassengers()
    initEmail()
    initEarthquakes()  # Add this line
    
# Backup the old database
def backup_database(db_uri, backup_uri):
    """Backup the current database."""
    if backup_uri:
        db_path = db_uri.replace('sqlite:///', 'instance/')
        backup_path = backup_uri.replace('sqlite:///', 'instance/')
        shutil.copyfile(db_path, backup_path)
        print(f"Database backed up to {backup_path}")
    else:
        print("Backup not supported for production database.")

# Extract data from the existing database
def extract_data():
    data = {}
    with app.app_context():
        data['users'] = [user.read() for user in User.query.all()]
        data['sections'] = [section.read() for section in Section.query.all()]
        data['channels'] = [channel.read() for channel in Channel.query.all()]
        data['team_members'] = [team_member.read() for team_member in TeamMember.query.all()]
        data['titanic'] = [titanic.read() for titanic in TitanicModel.query.all()]
        data['passengers'] = [passenger.read() for passenger in Passenger.query.all()] 
    return data

# Save extracted data to JSON files
def save_data_to_json(data, directory='backup'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for table, records in data.items():
        with open(os.path.join(directory, f'{table}.json'), 'w') as f:
            json.dump(records, f)
    print(f"Data backed up to {directory} directory.")

# Load data from JSON files
def load_data_from_json(directory='backup'):
    data = {}
    for table in ['polls', 'users', 'sections', 'groups', 'channels', 'school_classes', 'votes', 'team_members', 'top_interests', 'chat', 'languages']:
        with open(os.path.join(directory, f'{table}.json'), 'r') as f:
            data[table] = json.load(f)
    return data

def restore_data(data):
    with app.app_context():
        users = User.restore(data['users'])
        _ = Section.restore(data['sections'])
        _ = Channel.restore(data['channels'])
        _ = TeamMember.restore(data['team_members'])
        _ = Passenger.restore(data['passengers'])
    print("Data restored to the new database.")

# Define a command to backup data
@custom_cli.command('backup_data')
def backup_data():
    data = extract_data()
    save_data_to_json(data)
    backup_database(app.config['SQLALCHEMY_DATABASE_URI'], app.config['SQLALCHEMY_BACKUP_URI'])

# Define a command to restore data
@custom_cli.command('restore_data')
def restore_data_command():
    data = load_data_from_json()
    restore_data(data)
    
# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)

# this runs the flask application on the development server
if __name__ == "__main__":
    monitor_thread = threading.Thread(target=fire_monitor_thread, daemon=True)
    earthquake_monitor = threading.Thread(target=earthquake_monitor_thread, daemon=True)
    
    monitor_thread.start()
    earthquake_monitor.start()
    
    try:
        preprocess_data("fire_archive.csv")
    except FileNotFoundError:
        logger.warning("fire_archive.csv not found. Historical fire data will not be available.")
    except Exception as e:
        logger.error(f"Error preprocessing fire data: {e}")
    
    try:
        preprocess_earthquake_data("earthquakes.csv")
    except Exception as e:
        logger.error(f"Error preprocessing earthquake data: {e}")
    
    app.run(debug=True, host="0.0.0.0", port="8505")