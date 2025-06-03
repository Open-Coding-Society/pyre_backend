# from flask import Blueprint, jsonify, request
# from flask_restful import Api, Resource
# from model.email import Email

# email_api = Blueprint('email_api', __name__, url_prefix='/api')

# api = Api(email_api)

# ### TODO: Implement on the Game Page after the game is finished to save the game.

# class EmailApi:
#     class _BULK_CRUD(Resource):
#         def get(self):
#             emails = Email.query.all()
#             json_ready = []
#             for email in emails:
#                 email_data = email.read()
#                 json_ready.append(email_data)
#             return jsonify(json_ready)

#     class _CRUD(Resource):
#         def post(self):
#             body = request.get_json()
#             date = body['date']
#             email_id = body['email_id']
#             subject = body['subject']
#             sender = body['sender']
#             description = body['description']
#             hazard_type = body['hazard_type']
#             location = body['location']

#             ### TODO: Add error handling error for various input fields
#             email_obj = Email(date=date, email_id=email_id, subject=subject, sender=sender, description=description, hazard_type=hazard_type, location=location)
#             email_obj.create()
#             # if not pgn:  # failure returns error message
#             #     return {'message': f'Processed {body_pgn}, either a format error or User ID {body_id} is duplicate'}, 400

#             return jsonify(email_obj.read())

#         def delete(self):
#             body = request.get_json()
#             email = Email.query.get(body['id'])
#             if not email:
#                 return {'message': 'Crowdsourced email not found'}, 404
#             email.delete()
#             return jsonify({"message": "Email deleted"})

# api.add_resource(EmailApi._BULK_CRUD, '/emails')
# api.add_resource(EmailApi._CRUD, '/email') 