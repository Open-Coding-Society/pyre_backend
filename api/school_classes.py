from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from __init__ import db
from model.school_classes import SchoolClass

# Define the Blueprint for the SchoolClass API
school_class_api = Blueprint('school_class_api', __name__, url_prefix='/api')

# Connect the Api object to the Blueprint
api = Api(school_class_api)

class SchoolClassAPI:
    """
    Define the API CRUD endpoints for the SchoolClass model.
    """

    class _CRUD(Resource):
        def post(self):
            """
            Create a new school class.
            """
            data = request.get_json()

            # Validate required fields
            if not data or 'subject' not in data or 'teacher' not in data:
                return {'message': 'Subject and teacher(s) are required'}, 400

            # Create a new SchoolClass object
            school_class = SchoolClass(
                subject=data['subject'],
                teacher=data['teacher'] if isinstance(data['teacher'], list) else [data['teacher']],
                building=data.get('building')
            )

            # Save the school class using the ORM method
            try:
                school_class.create()
                return jsonify(school_class.read())
            except Exception as e:
                return {'message': f'Error creating school class: {e}'}, 500

        def get(self):
            """
            Retrieve all school classes or a specific class by ID.
            """
            class_id = request.args.get('id')

            if class_id:
                # Get a specific school class by ID
                school_class = SchoolClass.query.get(class_id)
                if not school_class:
                    return {'message': 'School class not found'}, 404
                return jsonify(school_class.read())

            # Get all school classes
            classes = SchoolClass.query.all()
            return jsonify([school_class.read() for school_class in classes])

        def put(self):
            """
            Update an existing school class by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for updating a class'}, 400

            # Find the school class by ID
            school_class = SchoolClass.query.get(data['id'])
            if not school_class:
                return {'message': 'School class not found'}, 404

            # Update the school class
            try:
                school_class.update(data)
                return jsonify(school_class.read())
            except Exception as e:
                return {'message': f'Error updating school class: {e}'}, 500

        def delete(self):
            """
            Delete a school class by ID.
            """
            data = request.get_json()

            if not data or 'id' not in data:
                return {'message': 'ID is required for deleting a class'}, 400

            # Find the school class by ID
            school_class = SchoolClass.query.get(data['id'])
            if not school_class:
                return {'message': 'School class not found'}, 404

            # Delete the school class
            try:
                school_class.delete()
                return {'message': 'School class deleted successfully'}, 200
            except Exception as e:
                return {'message': f'Error deleting school class: {e}'}, 500

    class _BY_TEACHER(Resource):
        def get(self):
            """
            Retrieve all classes taught by a specific teacher.
            """
            teacher_name = request.args.get('teacher')

            if not teacher_name:
                return {'message': 'Teacher name is required'}, 400

            classes = SchoolClass.query.filter(SchoolClass.teacher.like(f"%{teacher_name}%")).all()
            if not classes:
                return {'message': 'No classes found for the specified teacher'}, 404

            return jsonify([school_class.read() for school_class in classes])

    # Map the resources to API endpoints
    api.add_resource(_CRUD, '/school_class')
    api.add_resource(_BY_TEACHER, '/school_class/teacher')
