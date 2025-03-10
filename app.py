from flask import Flask, jsonify
from flask_cors import CORS

# initialize a flask application (app)
app = Flask(__name__)
CORS(app, supports_credentials=True, origins='*')  # Allow all origins (*)

# ... your existing Flask

# add an api endpoint to flask app
@app.route('/api/yash')
def get_yash():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Yash",
        "LastName": "Parikh",
        "DOB": "July 31",
        "Residence": "Antartica",
        "Email": "yashp51875@stu.powayusd.com",
        "Owns_Cars": ["2024-McLaren-W1-HotWheels"]
    })

    return jsonify(InfoDb)

@app.route('/api/anvay')
def get_anvay():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Anvay",
        "LastName": "Vahia",
        "DOB": "January 29",
        "Residence": "North Pole",
        "Email": "anvayv22800@stu.powayusd.com",
        "Owns_Cars": ["2023 Tesla Model Y", "2022 Hyundai Palisade"]
    })
    
    return jsonify(InfoDb)

# add an api endpoint to flask app
@app.route('/api/manas')
def get_manas():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Manas",
        "LastName": "Goel",
        "DOB": "July 12",
        "Residence": "San Diego",
        "Email": "manasg67038@stu.powayusd.com",
        "Owns_Cars": ["2024-Tesla", "2024-Mercedes"]
    })

    return jsonify(InfoDb)

# add an api endpoint to flask app
@app.route('/api/adi')
def get_adi():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Aditya",
        "LastName": "Katre",
        "DOB": "January 19",
        "Residence": "North Carolina",
        "Email": "adityak21664@stu.powayusd.com",
        "Owns_Cars": ["2022 Tesla Model Y Long Range", "2018 BMW 328i"]
    })
    
    return jsonify(InfoDb)

@app.route('/api/mihir')
def get_mihir():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Mihir",
        "LastName": "Bapat",
        "DOB": "May 26",
        "Residence": "Shrewsbury, United Kingdom",
        "Email": "mihirb59967@stu.powayusd.com",
        "Owns_Cars": ["2022 Tesla Model Y Long Range All-Wheel Drive"]
    })
    
    return jsonify(InfoDb)

# add an api endpoint to flask app
@app.route('/api/pranav')
def get_pranav():
    # start a list, to be used like a information database
    InfoDb = []

    # add a row to list, an Info record
    InfoDb.append({
        "FirstName": "Pranav",
        "LastName": "Santhosh",
        "DOB": "May 12",
        "Residence": "California",
        "Email": "pranavs22638@stu.powayusd.com",
        "Owns_Cars": ["2023 Rivian SUV"]
    })

    return jsonify(InfoDb)

# add an HTML endpoint to flask app
@app.route('/')
def say_hello():
    html_content = """
    <html>
    <head>
        <title>Group Members</title>
    </head>
    <body>
        <h2>Group Members Data</h2>
        <table border="1" style="width: 100%; text-align: left;">
            <thead>
                <tr>
                    <th>First Name</th>
                    <th>Last Name</th>
                    <th>Date of Birth</th>
                    <th>Residence</th>
                    <th>Email</th>
                    <th>Owns Cars</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # List of API endpoints
    api_endpoints = [
        '/api/yash',
        '/api/anvay',
        '/api/manas',
        '/api/adi',
        '/api/mihir',
        '/api/pranav'
    ]
    
    # Fetch data from APIs and populate the table
    for endpoint in api_endpoints:
        # Fetch data from the endpoint
        try:
            response = app.test_client().get(endpoint)
            data = response.get_json()
            if data:  # Iterate over the data if it exists
                for member in data:
                    html_content += f"""
                    <tr>
                        <td>{member['FirstName']}</td>
                        <td>{member['LastName']}</td>
                        <td>{member['DOB']}</td>
                        <td>{member['Residence']}</td>
                        <td>{member['Email']}</td>
                        <td>{', '.join(member['Owns_Cars'])}</td>
                    </tr>
                    """
        except Exception as e:
            html_content += f"<tr><td colspan='6'>Error fetching data from {endpoint}: {str(e)}</td></tr>"
    
    # Close the table and body
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    return html_content

if __name__ == '__main__':
    # starts flask server on default port, http://127.0.0.1:3333
    app.run(port=3333)