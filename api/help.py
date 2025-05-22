import jwt
from flask import Blueprint, request, jsonify, current_app, Response, g
from flask_restful import Api, Resource
from datetime import datetime
from __init__ import app, db
from api.jwt_authorize import token_required
from model.user import User
from sqlalchemy import desc, asc, or_, and_
import re

"""
This Blueprint object is used to define APIs for the Help/FAQ system.
- Handles user help requests, admin responses, FAQ management, and history tracking
- Blueprint is used to modularize application files.
- This Blueprint is registered to the Flask app in main.py.
"""
help_api = Blueprint('help_api', __name__, url_prefix='/api')

"""
The Api object is connected to the Blueprint object to define the API endpoints.
"""
api = Api(help_api)

class HelpAPI:
    """
    Define the API CRUD endpoints for the Help/FAQ system.
    Features include:
    - User help requests with categories and priority
    - Admin response system
    - FAQ management
    - Search functionality
    - History tracking
    - Status management
    """
    
    class _HELP_REQUEST(Resource):
        @token_required()
        def post(self):
            """
            Create a new help request from a user.
            """
            data = request.get_json()
            current_user = g.current_user
            
            # Validate required fields
            if not data:
                return {'message': 'No input data provided'}, 400
            if 'title' not in data or not data['title'].strip():
                return {'message': 'Help request title is required'}, 400
            if 'description' not in data or not data['description'].strip():
                return {'message': 'Help request description is required'}, 400
                
            # Validate category
            valid_categories = ['technical', 'account', 'general', 'bug_report', 'feature_request', 'billing', 'other']
            category = data.get('category', 'general').lower()
            if category not in valid_categories:
                return {'message': f'Invalid category. Must be one of: {", ".join(valid_categories)}'}, 400
                
            # Validate priority
            valid_priorities = ['low', 'medium', 'high', 'urgent']
            priority = data.get('priority', 'medium').lower()
            if priority not in valid_priorities:
                return {'message': f'Invalid priority. Must be one of: {", ".join(valid_priorities)}'}, 400
            
            try:
                # Create help request
                from model.help import HelpRequest
                help_request = HelpRequest(
                    title=data['title'].strip(),
                    description=data['description'].strip(),
                    category=category,
                    priority=priority,
                    user_id=current_user.id,
                    contact_email=data.get('contact_email', current_user.email),
                    status='open'
                )
                help_request.create()
                
                # Log the creation
                from model.help import HelpHistory
                history = HelpHistory(
                    help_request_id=help_request.id,
                    action='created',
                    details=f'Help request created by {current_user.name}',
                    user_id=current_user.id
                )
                history.create()
                
                return jsonify({
                    'message': 'Help request created successfully',
                    'request_id': help_request.id,
                    'data': help_request.read()
                }), 201
                
            except Exception as e:
                return {'message': f'Error creating help request: {str(e)}'}, 500

        @token_required()
        def get(self):
            """
            Get help requests. Users see their own, admins see all.
            """
            current_user = g.current_user
            
            try:
                from model.help import HelpRequest
                
                # Get query parameters
                status = request.args.get('status', 'all')
                category = request.args.get('category', 'all')
                priority = request.args.get('priority', 'all')
                page = int(request.args.get('page', 1))
                per_page = min(int(request.args.get('per_page', 10)), 100)
                sort_by = request.args.get('sort_by', 'created_date')
                sort_order = request.args.get('sort_order', 'desc')
                
                # Build query
                if current_user.role == 'Admin':
                    query = HelpRequest.query
                else:
                    query = HelpRequest.query.filter_by(user_id=current_user.id)
                
                # Apply filters
                if status != 'all':
                    query = query.filter_by(status=status)
                if category != 'all':
                    query = query.filter_by(category=category)
                if priority != 'all':
                    query = query.filter_by(priority=priority)
                    
                # Apply sorting
                if hasattr(HelpRequest, sort_by):
                    if sort_order.lower() == 'asc':
                        query = query.order_by(asc(getattr(HelpRequest, sort_by)))
                    else:
                        query = query.order_by(desc(getattr(HelpRequest, sort_by)))
                
                # Paginate
                help_requests = query.paginate(
                    page=page, per_page=per_page, error_out=False
                )
                
                return jsonify({
                    'requests': [req.read() for req in help_requests.items],
                    'pagination': {
                        'page': page,
                        'pages': help_requests.pages,
                        'per_page': per_page,
                        'total': help_requests.total,
                        'has_next': help_requests.has_next,
                        'has_prev': help_requests.has_prev
                    }
                })
                
            except Exception as e:
                return {'message': f'Error retrieving help requests: {str(e)}'}, 500

    class _HELP_REQUEST_DETAIL(Resource):
        @token_required()
        def get(self, request_id):
            """
            Get a specific help request with full details including history.
            """
            current_user = g.current_user
            
            try:
                from model.help import HelpRequest, HelpHistory
                
                help_request = HelpRequest.query.get(request_id)
                if not help_request:
                    return {'message': 'Help request not found'}, 404
                    
                # Check permissions
                if current_user.role != 'Admin' and help_request.user_id != current_user.id:
                    return {'message': 'Unauthorized to view this help request'}, 403
                    
                # Get history
                history = HelpHistory.query.filter_by(help_request_id=request_id).order_by(desc(HelpHistory.created_date)).all()
                
                response_data = help_request.read()
                response_data['history'] = [h.read() for h in history]
                
                return jsonify(response_data)
                
            except Exception as e:
                return {'message': f'Error retrieving help request: {str(e)}'}, 500

        @token_required()
        def put(self, request_id):
            """
            Update a help request (admin only for status/assignment, user for their own content).
            """
            current_user = g.current_user
            data = request.get_json()
            
            if not data:
                return {'message': 'No input data provided'}, 400
                
            try:
                from model.help import HelpRequest, HelpHistory
                
                help_request = HelpRequest.query.get(request_id)
                if not help_request:
                    return {'message': 'Help request not found'}, 404
                    
                # Check permissions
                if current_user.role != 'Admin' and help_request.user_id != current_user.id:
                    return {'message': 'Unauthorized to update this help request'}, 403
                
                changes = []
                
                # Admin can update status, priority, assignment
                if current_user.role == 'Admin':
                    if 'status' in data:
                        valid_statuses = ['open', 'in_progress', 'resolved', 'closed', 'on_hold']
                        if data['status'] in valid_statuses:
                            old_status = help_request.status
                            help_request.status = data['status']
                            changes.append(f'Status changed from {old_status} to {data["status"]}')
                            
                    if 'priority' in data:
                        valid_priorities = ['low', 'medium', 'high', 'urgent']
                        if data['priority'] in valid_priorities:
                            old_priority = help_request.priority
                            help_request.priority = data['priority']
                            changes.append(f'Priority changed from {old_priority} to {data["priority"]}')
                            
                    if 'assigned_to_id' in data:
                        admin_user = User.query.filter_by(id=data['assigned_to_id'], role='Admin').first()
                        if admin_user:
                            help_request.assigned_to_id = data['assigned_to_id']
                            changes.append(f'Assigned to {admin_user.name}')
                            
                    if 'admin_notes' in data:
                        help_request.admin_notes = data['admin_notes']
                        changes.append('Admin notes updated')
                
                # User can update their own request content (if not closed)
                if help_request.user_id == current_user.id and help_request.status not in ['resolved', 'closed']:
                    if 'title' in data and data['title'].strip():
                        help_request.title = data['title'].strip()
                        changes.append('Title updated')
                        
                    if 'description' in data and data['description'].strip():
                        help_request.description = data['description'].strip()
                        changes.append('Description updated')
                        
                    if 'contact_email' in data:
                        help_request.contact_email = data['contact_email']
                        changes.append('Contact email updated')
                
                if changes:
                    help_request.updated_date = datetime.now()
                    help_request.update()
                    
                    # Log the changes
                    history = HelpHistory(
                        help_request_id=request_id,
                        action='updated',
                        details='; '.join(changes),
                        user_id=current_user.id
                    )
                    history.create()
                    
                    return jsonify({
                        'message': 'Help request updated successfully',
                        'changes': changes,
                        'data': help_request.read()
                    })
                else:
                    return {'message': 'No valid changes provided'}, 400
                    
            except Exception as e:
                return {'message': f'Error updating help request: {str(e)}'}, 500

    class _ADMIN_RESPONSE(Resource):
        @token_required()
        def post(self, request_id):
            """
            Admin adds a response to a help request.
            """
            current_user = g.current_user
            
            if current_user.role != 'Admin':
                return {'message': 'Admin access required'}, 403
                
            data = request.get_json()
            if not data or 'response' not in data or not data['response'].strip():
                return {'message': 'Response text is required'}, 400
                
            try:
                from model.help import HelpRequest, HelpResponse, HelpHistory
                
                help_request = HelpRequest.query.get(request_id)
                if not help_request:
                    return {'message': 'Help request not found'}, 404
                
                # Create response
                response = HelpResponse(
                    help_request_id=request_id,
                    admin_id=current_user.id,
                    response_text=data['response'].strip(),
                    is_public=data.get('is_public', True)
                )
                response.create()
                
                # Update help request status if provided
                status_changed = False
                if 'new_status' in data and data['new_status']:
                    valid_statuses = ['open', 'in_progress', 'resolved', 'closed', 'on_hold']
                    if data['new_status'] in valid_statuses:
                        old_status = help_request.status
                        help_request.status = data['new_status']
                        help_request.update()
                        status_changed = True
                
                # Log the response
                details = f'Response added by {current_user.name}'
                if status_changed:
                    details += f'; Status changed to {data["new_status"]}'
                    
                history = HelpHistory(
                    help_request_id=request_id,
                    action='response_added',
                    details=details,
                    user_id=current_user.id
                )
                history.create()
                
                return jsonify({
                    'message': 'Response added successfully',
                    'response_id': response.id,
                    'data': response.read()
                }), 201
                
            except Exception as e:
                return {'message': f'Error adding response: {str(e)}'}, 500

    class _FAQ_MANAGEMENT(Resource):
        def get(self):
            """
            Get all FAQ items (public endpoint).
            """
            try:
                from model.help import FAQ
                
                category = request.args.get('category', 'all')
                search = request.args.get('search', '')
                
                query = FAQ.query.filter_by(is_active=True)
                
                if category != 'all':
                    query = query.filter_by(category=category)
                    
                if search:
                    search_term = f'%{search}%'
                    query = query.filter(
                        or_(
                            FAQ.question.ilike(search_term),
                            FAQ.answer.ilike(search_term),
                            FAQ.tags.ilike(search_term)
                        )
                    )
                
                faqs = query.order_by(FAQ.display_order, FAQ.created_date).all()
                
                # Group by category
                grouped_faqs = {}
                for faq in faqs:
                    if faq.category not in grouped_faqs:
                        grouped_faqs[faq.category] = []
                    grouped_faqs[faq.category].append(faq.read())
                
                return jsonify({
                    'faqs': grouped_faqs,
                    'total_count': len(faqs)
                })
                
            except Exception as e:
                return {'message': f'Error retrieving FAQs: {str(e)}'}, 500

        @token_required()
        def post(self):
            """
            Create a new FAQ item (admin only).
            """
            current_user = g.current_user
            
            if current_user.role != 'Admin':
                return {'message': 'Admin access required'}, 403
                
            data = request.get_json()
            if not data:
                return {'message': 'No input data provided'}, 400
                
            required_fields = ['question', 'answer', 'category']
            for field in required_fields:
                if field not in data or not data[field].strip():
                    return {'message': f'{field.title()} is required'}, 400
            
            try:
                from model.help import FAQ
                
                faq = FAQ(
                    question=data['question'].strip(),
                    answer=data['answer'].strip(),
                    category=data['category'].strip().lower(),
                    tags=data.get('tags', ''),
                    display_order=data.get('display_order', 0),
                    created_by_id=current_user.id
                )
                faq.create()
                
                return jsonify({
                    'message': 'FAQ created successfully',
                    'faq_id': faq.id,
                    'data': faq.read()
                }), 201
                
            except Exception as e:
                return {'message': f'Error creating FAQ: {str(e)}'}, 500

    class _FAQ_DETAIL(Resource):
        @token_required()
        def put(self, faq_id):
            """
            Update an FAQ item (admin only).
            """
            current_user = g.current_user
            
            if current_user.role != 'Admin':
                return {'message': 'Admin access required'}, 403
                
            data = request.get_json()
            if not data:
                return {'message': 'No input data provided'}, 400
                
            try:
                from model.help import FAQ
                
                faq = FAQ.query.get(faq_id)
                if not faq:
                    return {'message': 'FAQ not found'}, 404
                
                # Update fields
                if 'question' in data and data['question'].strip():
                    faq.question = data['question'].strip()
                if 'answer' in data and data['answer'].strip():
                    faq.answer = data['answer'].strip()
                if 'category' in data and data['category'].strip():
                    faq.category = data['category'].strip().lower()
                if 'tags' in data:
                    faq.tags = data['tags']
                if 'display_order' in data:
                    faq.display_order = data['display_order']
                if 'is_active' in data:
                    faq.is_active = bool(data['is_active'])
                    
                faq.updated_date = datetime.now()
                faq.update()
                
                return jsonify({
                    'message': 'FAQ updated successfully',
                    'data': faq.read()
                })
                
            except Exception as e:
                return {'message': f'Error updating FAQ: {str(e)}'}, 500

        @token_required()
        def delete(self, faq_id):
            """
            Delete an FAQ item (admin only).
            """
            current_user = g.current_user
            
            if current_user.role != 'Admin':
                return {'message': 'Admin access required'}, 403
                
            try:
                from model.help import FAQ
                
                faq = FAQ.query.get(faq_id)
                if not faq:
                    return {'message': 'FAQ not found'}, 404
                
                faq.delete()
                
                return jsonify({'message': 'FAQ deleted successfully'})
                
            except Exception as e:
                return {'message': f'Error deleting FAQ: {str(e)}'}, 500

    class _SEARCH(Resource):
        def get(self):
            """
            Search through help requests and FAQs.
            """
            query = request.args.get('q', '').strip()
            if not query:
                return {'message': 'Search query is required'}, 400
                
            try:
                from model.help import FAQ, HelpRequest
                
                # Search FAQs (public)
                faq_results = FAQ.query.filter(
                    and_(
                        FAQ.is_active == True,
                        or_(
                            FAQ.question.ilike(f'%{query}%'),
                            FAQ.answer.ilike(f'%{query}%'),
                            FAQ.tags.ilike(f'%{query}%')
                        )
                    )
                ).limit(10).all()
                
                results = {
                    'faqs': [faq.read() for faq in faq_results],
                    'help_requests': []
                }
                
                # Search help requests (authenticated users only)
                if hasattr(g, 'current_user') and g.current_user:
                    current_user = g.current_user
                    help_query = HelpRequest.query
                    
                    if current_user.role != 'Admin':
                        help_query = help_query.filter_by(user_id=current_user.id)
                    
                    help_results = help_query.filter(
                        or_(
                            HelpRequest.title.ilike(f'%{query}%'),
                            HelpRequest.description.ilike(f'%{query}%')
                        )
                    ).limit(10).all()
                    
                    results['help_requests'] = [req.read() for req in help_results]
                
                return jsonify(results)
                
            except Exception as e:
                return {'message': f'Error performing search: {str(e)}'}, 500

    class _STATISTICS(Resource):
        @token_required()
        def get(self):
            """
            Get help system statistics (admin only).
            """
            current_user = g.current_user
            
            if current_user.role != 'Admin':
                return {'message': 'Admin access required'}, 403
                
            try:
                from model.help import HelpRequest, HelpResponse, FAQ
                from sqlalchemy import func
                
                # Request statistics
                total_requests = HelpRequest.query.count()
                open_requests = HelpRequest.query.filter_by(status='open').count()
                in_progress_requests = HelpRequest.query.filter_by(status='in_progress').count()
                resolved_requests = HelpRequest.query.filter_by(status='resolved').count()
                
                # Category breakdown
                category_stats = db.session.query(
                    HelpRequest.category,
                    func.count(HelpRequest.id).label('count')
                ).group_by(HelpRequest.category).all()
                
                # Priority breakdown
                priority_stats = db.session.query(
                    HelpRequest.priority,
                    func.count(HelpRequest.id).label('count')
                ).group_by(HelpRequest.priority).all()
                
                # Response statistics
                total_responses = HelpResponse.query.count()
                avg_response_time = db.session.query(
                    func.avg(
                        func.julianday(HelpResponse.created_date) - 
                        func.julianday(HelpRequest.created_date)
                    )
                ).join(HelpRequest).scalar()
                
                # FAQ statistics
                total_faqs = FAQ.query.filter_by(is_active=True).count()
                
                return jsonify({
                    'requests': {
                        'total': total_requests,
                        'open': open_requests,
                        'in_progress': in_progress_requests,
                        'resolved': resolved_requests,
                        'by_category': {cat: count for cat, count in category_stats},
                        'by_priority': {pri: count for pri, count in priority_stats}
                    },
                    'responses': {
                        'total': total_responses,
                        'avg_response_time_days': round(avg_response_time, 2) if avg_response_time else 0
                    },
                    'faqs': {
                        'total': total_faqs
                    }
                })
                
            except Exception as e:
                return {'message': f'Error retrieving statistics: {str(e)}'}, 500

    """
    Map all the resource classes to their respective API endpoints.
    """
    # Help request endpoints
    api.add_resource(_HELP_REQUEST, '/help/requests')
    api.add_resource(_HELP_REQUEST_DETAIL, '/help/requests/<int:request_id>')
    api.add_resource(_ADMIN_RESPONSE, '/help/requests/<int:request_id>/response')
    
    # FAQ endpoints
    api.add_resource(_FAQ_MANAGEMENT, '/help/faqs')
    api.add_resource(_FAQ_DETAIL, '/help/faqs/<int:faq_id>')
    
    # Utility endpoints
    api.add_resource(_SEARCH, '/help/search')
    api.add_resource(_STATISTICS, '/help/statistics')