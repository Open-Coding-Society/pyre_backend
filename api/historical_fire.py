from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
import numpy as np
import pandas as pd
from model.historical_fire import FireDataAnalysisAdvancedRegressionModel, FireDataPolynomialRegressionModel
from __init__ import app, db

historical_fire_api = Blueprint('historical_fire_api', __name__,
                   url_prefix='/api/historical-fire')

api = Api(historical_fire_api)

class FireDataAnalysisAdvancedRegressionAPI:
    class _Analyze(Resource):
        def post(self):
            """
            Semantics: POST request to analyze fire data for a specific time period
            
            Expected JSON payload:
            {
                "year": 2020,        # Optional: specific year (null for all years)
                "month": 8,          # Optional: specific month 1-12 (null for all months)
                "analysis_type": "comprehensive"  # Options: "comprehensive", "time_series", "spatial", "statistics", "animation"
            }
            
            Returns comprehensive analysis including:
            - Time series forecasting with Prophet
            - Spatial clustering analysis
            - Statistical summaries
            - CSV data for frontend visualization
            - Base64 encoded plots/graphs
            - Animation data for dynamic visualizations
            """
            try:
                # Get the analysis parameters from the request
                analysis_data = request.get_json()
                
                # Validate input data
                if not analysis_data:
                    return {"error": "No analysis parameters provided"}, 400
                
                # Extract parameters
                year = analysis_data.get('year', None)
                month = analysis_data.get('month', None) 
                analysis_type = analysis_data.get('analysis_type', 'comprehensive')
                
                # Validate year and month ranges
                if year is not None and (year < 2000 or year > 2030):
                    return {"error": "Year must be between 2000 and 2030"}, 400
                
                if month is not None and (month < 1 or month > 12):
                    return {"error": "Month must be between 1 and 12"}, 400
                
                # Get the singleton instance of the FireDataAnalysisAdvancedRegressionModel
                analysisModel = FireDataAnalysisAdvancedRegressionModel.get_instance()
                
                # Ensure data is loaded
                if analysisModel.data is None:
                    load_result = analysisModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Run the appropriate analysis based on type
                if analysis_type == "comprehensive":
                    response = analysisModel.run_comprehensive_analysis(year, month)
                elif analysis_type == "time_series":
                    response = analysisModel.generate_time_series_analysis(year, month)
                elif analysis_type == "spatial":
                    response = analysisModel.generate_spatial_analysis(year, month)
                elif analysis_type == "statistics":
                    response = analysisModel.generate_statistical_summary(year, month)
                elif analysis_type == "animation":
                    response = analysisModel.generate_animated_visualization(year, month)
                else:
                    return {"error": "Invalid analysis_type. Choose from: comprehensive, time_series, spatial, statistics, animation"}, 400
                
                # Return the response as JSON
                if response.get("status") == "success":
                    return jsonify(response)
                else:
                    return {"error": response.get("message", "Analysis failed")}, 500
                    
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _DataSummary(Resource):
        def get(self):
            """
            GET request to retrieve basic data summary and available periods
            
            Returns:
            - Available years in dataset
            - Available months per year
            - Total fire count
            - Data range information
            """
            try:
                # Get the singleton instance
                analysisModel = FireDataAnalysisAdvancedRegressionModel.get_instance()
                
                # Ensure data is loaded
                if analysisModel.data is None:
                    load_result = analysisModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                data = analysisModel.data
                
                # Generate summary information
                summary = {
                    "status": "success",
                    "total_records": int(len(data)),
                    "date_range": {
                        "start": data['acq_date'].min().strftime('%Y-%m-%d'),
                        "end": data['acq_date'].max().strftime('%Y-%m-%d')
                    },
                    "available_years": sorted(data['year'].unique().tolist()),
                    "available_months": sorted(data['month'].unique().tolist()),
                    "records_per_year": data.groupby('year').size().to_dict(),
                    "records_per_month": data.groupby('month').size().to_dict()
                }
                
                return jsonify(summary)
                
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _ExportData(Resource):
        def post(self):
            """
            POST request to export filtered data as CSV
            
            Expected JSON payload:
            {
                "year": 2020,        # Optional: specific year
                "month": 8,          # Optional: specific month
                "format": "csv"      # Export format (currently only CSV supported)
            }
            
            Returns filtered data in the requested format
            """
            try:
                # Get export parameters
                export_data = request.get_json()
                
                if not export_data:
                    return {"error": "No export parameters provided"}, 400
                
                year = export_data.get('year', None)
                month = export_data.get('month', None)
                export_format = export_data.get('format', 'csv')
                
                if export_format != 'csv':
                    return {"error": "Currently only CSV format is supported"}, 400
                
                # Get the singleton instance
                analysisModel = FireDataAnalysisAdvancedRegressionModel.get_instance()
                
                # Ensure data is loaded
                if analysisModel.data is None:
                    load_result = analysisModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Filter data
                filtered_data = analysisModel.filter_data_by_period(year, month)
                
                if filtered_data is None or len(filtered_data) == 0:
                    return {"error": "No data available for the specified period"}, 404
                
                # Convert to CSV format (as JSON for API response)
                csv_data = filtered_data.to_dict('records')
                
                response = {
                    "status": "success",
                    "format": "csv",
                    "record_count": len(csv_data),
                    "period": f"{year or 'All Years'} - {month or 'All Months'}",
                    "data": csv_data
                }
                
                return jsonify(response)
                
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _ModelStatus(Resource):
        def get(self):
            """
            GET request to check model status and health
            
            Returns model status, data loading status, and system information
            """
            try:
                analysisModel = FireDataAnalysisAdvancedRegressionModel.get_instance()
                
                status = {
                    "status": "healthy",
                    "model_loaded": analysisModel is not None,
                    "data_loaded": analysisModel.data is not None,
                    "prophet_model_trained": analysisModel.prophet_model is not None,
                    "clustering_model_available": analysisModel.clustering_model is not None
                }
                
                if analysisModel.data is not None:
                    status["data_info"] = {
                        "total_records": len(analysisModel.data),
                        "columns": list(analysisModel.data.columns),
                        "memory_usage_mb": round(analysisModel.data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                    }
                
                return jsonify(status)
                
            except Exception as e:
                return {"error": f"Model status check failed: {str(e)}"}, 500

    api.add_resource(_Analyze, '/advanced/analyze')
    api.add_resource(_DataSummary, '/advanced/summary') 
    api.add_resource(_ExportData, '/advanced/export')
    api.add_resource(_ModelStatus, '/advanced/status')

class FireDataPolynomialRegressionAPI:
    
    class _AnalyzePolynomial(Resource):
        def post(self):
            """
            Semantics: POST request to perform polynomial regression analysis on fire data
            
            Expected JSON payload:
            {
                "year": 2020,              # Optional: specific year (null for all years)
                "month": 8,                # Optional: specific month 1-12 (null for all months)
                "degree": 4,               # Optional: polynomial degree (default: 4)
                "analysis_type": "comprehensive"  # Options: "comprehensive", "trend_only", "degree_comparison"
            }
            
            Returns polynomial regression analysis including:
            - Trend analysis with future predictions
            - Residuals analysis
            - Model performance metrics (MSE, R², MAE)
            - Polynomial coefficients visualization
            - CSV data for frontend visualization
            - Base64 encoded plots
            """
            try:
                # Get the analysis parameters from the request
                analysis_data = request.get_json()
                
                # Validate input data
                if not analysis_data:
                    return {"error": "No analysis parameters provided"}, 400
                
                # Extract parameters
                year = analysis_data.get('year', None)
                month = analysis_data.get('month', None)
                degree = analysis_data.get('degree', 4)
                analysis_type = analysis_data.get('analysis_type', 'comprehensive')
                
                # Validate parameters
                if year is not None and (year < 2000 or year > 2030):
                    return {"error": "Year must be between 2000 and 2030"}, 400
                
                if month is not None and (month < 1 or month > 12):
                    return {"error": "Month must be between 1 and 12"}, 400
                
                if degree < 1 or degree > 10:
                    return {"error": "Polynomial degree must be between 1 and 10"}, 400
                
                # Get the singleton instance of the FireDataPolynomialRegressionModel
                polyModel = FireDataPolynomialRegressionModel.get_instance()
                
                # Ensure data is loaded
                if polyModel.data is None:
                    load_result = polyModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Run the appropriate analysis based on type
                if analysis_type == "comprehensive":
                    response = polyModel.run_comprehensive_polynomial_analysis(year, month, degree)
                elif analysis_type == "trend_only":
                    response = polyModel.generate_polynomial_analysis(year, month, degree)
                elif analysis_type == "degree_comparison":
                    degrees = analysis_data.get('degrees', [2, 3, 4, 5])
                    response = polyModel.compare_polynomial_degrees(year, month, degrees)
                else:
                    return {"error": "Invalid analysis_type. Choose from: comprehensive, trend_only, degree_comparison"}, 400
                
                # Return the response as JSON
                if response.get("status") == "success":
                    return jsonify(response)
                else:
                    return {"error": response.get("message", "Polynomial analysis failed")}, 500
                    
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _OptimizeDegree(Resource):
        def post(self):
            """
            POST request to find optimal polynomial degree for given data
            
            Expected JSON payload:
            {
                "year": 2020,                    # Optional: specific year
                "month": 8,                      # Optional: specific month
                "max_degree": 8,                 # Optional: maximum degree to test (default: 8)
                "min_degree": 2                  # Optional: minimum degree to test (default: 2)
            }
            
            Returns the optimal polynomial degree based on validation metrics
            """
            try:
                analysis_data = request.get_json()
                
                if not analysis_data:
                    return {"error": "No optimization parameters provided"}, 400
                
                year = analysis_data.get('year', None)
                month = analysis_data.get('month', None)
                min_degree = analysis_data.get('min_degree', 2)
                max_degree = analysis_data.get('max_degree', 8)
                
                # Validate parameters
                if min_degree >= max_degree:
                    return {"error": "min_degree must be less than max_degree"}, 400
                
                if max_degree > 15:
                    return {"error": "max_degree should not exceed 15 to avoid overfitting"}, 400
                
                # Get model instance
                polyModel = FireDataPolynomialRegressionModel.get_instance()
                
                # Ensure data is loaded
                if polyModel.data is None:
                    load_result = polyModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Generate degree range
                degrees = list(range(min_degree, max_degree + 1))
                
                # Run comparison
                response = polyModel.compare_polynomial_degrees(year, month, degrees)
                
                if response.get("status") == "success":
                    return jsonify(response)
                else:
                    return {"error": response.get("error", "Degree optimization failed")}, 500
                    
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _PredictFuture(Resource):
        def post(self):
            """
            POST request to generate future predictions using polynomial regression
            
            Expected JSON payload:
            {
                "year": 2020,              # Optional: specific year for training
                "month": 8,                # Optional: specific month for training
                "degree": 4,               # Optional: polynomial degree
                "periods": 12,             # Optional: number of future periods to predict
                "period_type": "months"    # Optional: "months" or "days"
            }
            
            Returns future fire count predictions
            """
            try:
                analysis_data = request.get_json()
                
                if not analysis_data:
                    return {"error": "No prediction parameters provided"}, 400
                
                year = analysis_data.get('year', None)
                month = analysis_data.get('month', None)
                degree = analysis_data.get('degree', 4)
                periods = analysis_data.get('periods', 12)
                period_type = analysis_data.get('period_type', 'months')
                
                # Validate parameters
                if periods <= 0 or periods > 60:
                    return {"error": "Periods must be between 1 and 60"}, 400
                
                if period_type not in ['months', 'days']:
                    return {"error": "period_type must be 'months' or 'days'"}, 400
                
                # Get model instance
                polyModel = FireDataPolynomialRegressionModel.get_instance()
                
                # Ensure data is loaded
                if polyModel.data is None:
                    load_result = polyModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Get filtered data and train model
                filtered_data = polyModel.filter_data_by_period(year, month)
                if filtered_data is None or len(filtered_data) == 0:
                    return {"error": "No data available for the specified period"}, 404
                
                model_result = polyModel.train_polynomial_model(filtered_data, degree)
                if "error" in model_result:
                    return {"error": model_result["error"]}, 500
                
                poly_model = model_result["model"]
                X = model_result["X"]
                
                # Generate future timestamps
                last_timestamp = X.max()
                if period_type == 'months':
                    time_increment = 365.25 * 24 * 3600 / 12  # seconds per month
                else:  # days
                    time_increment = 24 * 3600  # seconds per day
                
                future_timestamps = np.array([last_timestamp + (i + 1) * time_increment 
                                            for i in range(periods)]).reshape(-1, 1)
                
                # Generate predictions
                future_predictions = poly_model.predict(future_timestamps)
                
                # Create future dates
                last_date = pd.to_datetime(model_result["data"]['year_month_str'].max())
                if period_type == 'months':
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                               periods=periods, freq='M')
                else:
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                               periods=periods, freq='D')
                
                # Prepare response
                predictions = []
                for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
                    predictions.append({
                        "period": date.strftime('%Y-%m-%d' if period_type == 'days' else '%Y-%m'),
                        "predicted_fire_count": float(max(0, pred)),  # Ensure non-negative
                        "timestamp": float(future_timestamps[i][0])
                    })
                
                response = {
                    "status": "success",
                    "predictions": predictions,
                    "model_info": {
                        "polynomial_degree": degree,
                        "training_period": f"{year or 'All Years'} - {month or 'All Months'}",
                        "prediction_type": period_type,
                        "periods_predicted": periods
                    },
                    "summary": {
                        "avg_predicted_count": float(np.mean(future_predictions)),
                        "max_predicted_count": float(np.max(future_predictions)),
                        "min_predicted_count": float(max(0, np.min(future_predictions))),
                        "trend": "increasing" if future_predictions[-1] > future_predictions[0] else "decreasing"
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    class _ModelMetrics(Resource):
        def post(self):
            """
            POST request to get detailed model performance metrics
            
            Expected JSON payload:
            {
                "year": 2020,              # Optional: specific year
                "month": 8,                # Optional: specific month  
                "degree": 4                # Optional: polynomial degree
            }
            
            Returns detailed performance metrics and diagnostics
            """
            try:
                analysis_data = request.get_json()
                
                if not analysis_data:
                    return {"error": "No metrics parameters provided"}, 400
                
                year = analysis_data.get('year', None)
                month = analysis_data.get('month', None)
                degree = analysis_data.get('degree', 4)
                
                # Get model instance
                polyModel = FireDataPolynomialRegressionModel.get_instance()
                
                # Ensure data is loaded
                if polyModel.data is None:
                    load_result = polyModel.load_data()
                    if load_result["status"] == "error":
                        return {"error": f"Failed to load data: {load_result['message']}"}, 500
                
                # Run analysis to get metrics
                result = polyModel.generate_polynomial_analysis(year, month, degree)
                
                if result.get("status") == "success":
                    metrics_response = {
                        "status": "success",
                        "model_metrics": result.get("model_metrics", {}),
                        "summary": result.get("summary", {}),
                        "residuals_plot": result.get("plots", {}).get("residuals"),
                        "coefficients_plot": result.get("plots", {}).get("coefficients")
                    }
                    return jsonify(metrics_response)
                else:
                    return {"error": result.get("error", "Metrics calculation failed")}, 500
                    
            except Exception as e:
                return {"error": f"Internal server error: {str(e)}"}, 500

    api.add_resource(_AnalyzePolynomial, '/polynomial/analyze')
    api.add_resource(_OptimizeDegree, '/polynomial/optimize-degree')
    api.add_resource(_PredictFuture, '/polynomial/predict')
    api.add_resource(_ModelMetrics, '/polynomial/metrics')

