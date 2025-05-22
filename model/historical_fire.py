import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import io
import base64
import json
from datetime import datetime, timedelta
from __init__ import app, db

class FireDataAnalysisAdvancedRegressionModel:
    _instance = None

    @staticmethod
    def get_instance():
        if FireDataAnalysisAdvancedRegressionModel._instance is None:
            FireDataAnalysisAdvancedRegressionModel()
        return FireDataAnalysisAdvancedRegressionModel._instance

    def __init__(self):
        if FireDataAnalysisAdvancedRegressionModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            FireDataAnalysisAdvancedRegressionModel._instance = self
            self.data = None
            self.prophet_model = None
            self.clustering_model = None
            self.scaler = StandardScaler()
            
    def load_data(self, file_path="fire_archive.csv"):
        """Load and preprocess the fire data"""
        try:
            self.data = pd.read_csv(file_path, parse_dates=['acq_date'])
            # Filter data to reasonable range
            self.data = self.data[(self.data['acq_date'].dt.year >= 2015) & 
                                 (self.data['acq_date'].dt.year <= 2022)]
            
            # Add additional time features
            self.data['year'] = self.data['acq_date'].dt.year
            self.data['month'] = self.data['acq_date'].dt.month
            self.data['day_of_year'] = self.data['acq_date'].dt.dayofyear
            self.data['year_month'] = self.data['acq_date'].dt.to_period('M')
            
            return {"status": "success", "message": "Data loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def filter_data_by_period(self, year=None, month=None):
        """Filter data by specific year and/or month"""
        if self.data is None:
            return None
            
        filtered_data = self.data.copy()
        
        if year is not None:
            filtered_data = filtered_data[filtered_data['year'] == year]
        
        if month is not None:
            filtered_data = filtered_data[filtered_data['month'] == month]
            
        return filtered_data

    def generate_time_series_analysis(self, year=None, month=None):
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Prepare data for Prophet
            if month is not None:
                # Daily aggregation for specific month
                daily_fires = filtered_data.groupby('acq_date').size().reset_index(name='fire_count')
                daily_fires['ds'] = daily_fires['acq_date']
                daily_fires['y'] = daily_fires['fire_count']
                freq = 'D'
                periods = 30  # Forecast 30 days
            else:
                # Monthly aggregation
                monthly_fires = filtered_data.groupby('year_month').size().reset_index(name='fire_count')
                monthly_fires['ds'] = pd.to_datetime(monthly_fires['year_month'].astype(str))
                monthly_fires['y'] = monthly_fires['fire_count']
                daily_fires = monthly_fires
                freq = 'M'
                periods = 12  # Forecast 12 months

            # Train Prophet model
            model = Prophet(yearly_seasonality=True, daily_seasonality=(month is not None))
            model.fit(daily_fires[['ds', 'y']])
            
            # Generate forecast
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            
            # Generate plots
            plots = {}
            
            # Forecast plot
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            model.plot(forecast, ax=ax1)
            ax1.set_title(f"Fire Count Forecast - {year or 'All Years'} {month or 'All Months'}")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Fire Count")
            plots['forecast'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Components plot
            fig2 = model.plot_components(forecast)
            fig2.suptitle(f"Forecast Components - {year or 'All Years'} {month or 'All Months'}")
            plots['components'] = self._fig_to_base64(fig2)
            plt.close(fig2)
            
            # Actual vs Predicted
            merged = pd.merge(daily_fires, forecast[['ds', 'yhat']], on='ds', how='left')
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(merged['ds'], merged['y'], label='Actual', linewidth=2)
            ax3.plot(merged['ds'], merged['yhat'], label='Predicted', linewidth=2, alpha=0.8)
            ax3.set_title(f"Actual vs Predicted Fire Counts - {year or 'All Years'} {month or 'All Months'}")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Fire Count")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plots['comparison'] = self._fig_to_base64(fig3)
            plt.close(fig3)
            
            # Generate CSV data
            csv_data = {
                'historical': daily_fires.to_dict('records'),
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records')
            }
            
            return {
                "status": "success",
                "plots": plots,
                "csv_data": csv_data,
                "summary": {
                    "total_fires": int(filtered_data.shape[0]),
                    "avg_daily_fires": float(daily_fires['y'].mean()),
                    "max_daily_fires": int(daily_fires['y'].max()),
                    "forecast_avg": float(forecast['yhat'].tail(periods).mean())
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_spatial_analysis(self, year=None, month=None):
        """Generate spatial clustering analysis"""
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Check if latitude and longitude columns exist
            if 'latitude' not in filtered_data.columns or 'longitude' not in filtered_data.columns:
                return {"error": "Spatial data (latitude/longitude) not available"}

            # Prepare spatial data
            spatial_data = filtered_data[['latitude', 'longitude']].dropna()
            if len(spatial_data) < 10:
                return {"error": "Insufficient spatial data for clustering"}

            # Perform clustering
            n_clusters = min(8, len(spatial_data) // 5)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(spatial_data)
            
            # Create spatial analysis plots
            plots = {}
            
            # Cluster map
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            scatter = ax1.scatter(spatial_data['longitude'], spatial_data['latitude'], 
                                c=clusters, cmap='viridis', alpha=0.6, s=20)
            ax1.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            ax1.set_title(f"Fire Hotspot Clusters - {year or 'All Years'} {month or 'All Months'}")
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")
            ax1.legend()
            plt.colorbar(scatter, ax=ax1, label='Cluster')
            plots['clusters'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_fires = spatial_data[cluster_mask]
                cluster_stats.append({
                    'cluster_id': int(i),
                    'fire_count': int(cluster_mask.sum()),
                    'center_lat': float(kmeans.cluster_centers_[i, 0]),
                    'center_lon': float(kmeans.cluster_centers_[i, 1]),
                    'avg_lat': float(cluster_fires['latitude'].mean()),
                    'avg_lon': float(cluster_fires['longitude'].mean())
                })
            
            return {
                "status": "success",
                "plots": plots,
                "cluster_data": cluster_stats,
                "summary": {
                    "total_fires": int(len(spatial_data)),
                    "n_clusters": n_clusters,
                    "largest_cluster": max(cluster_stats, key=lambda x: x['fire_count'])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_statistical_summary(self, year=None, month=None):
        """Generate comprehensive statistical analysis"""
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Generate summary statistics
            plots = {}
            
            # Monthly distribution
            monthly_counts = filtered_data.groupby('month').size()
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            monthly_counts.plot(kind='bar', ax=ax1, color='orangered', alpha=0.7)
            ax1.set_title(f"Fire Distribution by Month - {year or 'All Years'}")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Fire Count")
            ax1.grid(True, alpha=0.3)
            plots['monthly_distribution'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Yearly trend (if not filtered by specific year)
            if year is None:
                yearly_counts = filtered_data.groupby('year').size()
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                yearly_counts.plot(kind='line', ax=ax2, marker='o', linewidth=2, markersize=8, color='darkred')
                ax2.set_title("Fire Count Trend by Year")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Fire Count")
                ax2.grid(True, alpha=0.3)
                plots['yearly_trend'] = self._fig_to_base64(fig2)
                plt.close(fig2)
            
            # Generate summary statistics
            summary_stats = {
                "total_fires": int(len(filtered_data)),
                "period": f"{year or 'All Years'} - {month or 'All Months'}",
                "date_range": {
                    "start": filtered_data['acq_date'].min().strftime('%Y-%m-%d'),
                    "end": filtered_data['acq_date'].max().strftime('%Y-%m-%d')
                },
                "monthly_breakdown": monthly_counts.to_dict(),
                "peak_month": int(monthly_counts.idxmax()),
                "avg_fires_per_month": float(monthly_counts.mean())
            }
            
            if year is None:
                summary_stats["yearly_breakdown"] = filtered_data.groupby('year').size().to_dict()
            
            return {
                "status": "success",
                "plots": plots,
                "summary": summary_stats,
                "csv_data": filtered_data.groupby(['year', 'month']).size().reset_index(name='fire_count').to_dict('records')
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_animated_visualization(self, year=None, month=None):
        """Generate animated visualization data"""
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Prepare animation data
            if month is not None:
                # Daily animation for specific month
                daily_data = filtered_data.groupby('acq_date').size().reset_index(name='fire_count')
                daily_data['date_str'] = daily_data['acq_date'].dt.strftime('%Y-%m-%d')
                animation_data = daily_data[['date_str', 'fire_count']].to_dict('records')
            else:
                # Monthly animation
                monthly_data = filtered_data.groupby('year_month').size().reset_index(name='fire_count')
                monthly_data['period_str'] = monthly_data['year_month'].astype(str)
                animation_data = monthly_data[['period_str', 'fire_count']].to_dict('records')
            
            return {
                "status": "success",
                "animation_data": animation_data,
                "summary": {
                    "total_frames": len(animation_data),
                    "max_fires": max([d['fire_count'] for d in animation_data]),
                    "min_fires": min([d['fire_count'] for d in animation_data])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def run_comprehensive_analysis(self, year=None, month=None):
        """Run all analyses and return comprehensive results"""
        try:
            if self.data is None:
                load_result = self.load_data()
                if load_result["status"] == "error":
                    return load_result

            results = {
                "status": "success",
                "period": f"{year or 'All Years'} - {month or 'All Months'}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Run time series analysis
            ts_result = self.generate_time_series_analysis(year, month)
            if ts_result.get("status") == "success":
                results["time_series"] = ts_result
            else:
                results["time_series"] = {"error": ts_result.get("error")}
            
            # Run spatial analysis
            spatial_result = self.generate_spatial_analysis(year, month)
            if spatial_result.get("status") == "success":
                results["spatial"] = spatial_result
            else:
                results["spatial"] = {"error": spatial_result.get("error")}
            
            # Run statistical summary
            stats_result = self.generate_statistical_summary(year, month)
            if stats_result.get("status") == "success":
                results["statistics"] = stats_result
            else:
                results["statistics"] = {"error": stats_result.get("error")}
            
            # Run animation data generation
            anim_result = self.generate_animated_visualization(year, month)
            if anim_result.get("status") == "success":
                results["animation"] = anim_result
            else:
                results["animation"] = {"error": anim_result.get("error")}
            
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{image_base64}"

class FireDataPolynomialRegressionModel:
    _instance = None

    @staticmethod
    def get_instance():
        if FireDataPolynomialRegressionModel._instance is None:
            FireDataPolynomialRegressionModel()
        return FireDataPolynomialRegressionModel._instance

    def __init__(self):
        if FireDataPolynomialRegressionModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            FireDataPolynomialRegressionModel._instance = self
            self.data = None
            self.poly_model = None
            self.degree = 4  # Default polynomial degree
            
    def load_data(self, file_path="fire_archive.csv"):
        """Load and preprocess the fire data"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            
            self.data = pd.read_csv(file_path, parse_dates=['acq_date'])
            # Filter data to reasonable range
            self.data = self.data[(self.data['acq_date'].dt.year >= 2015) & 
                                 (self.data['acq_date'].dt.year <= 2022)]
            
            # Add time features
            self.data['year'] = self.data['acq_date'].dt.year
            self.data['month'] = self.data['acq_date'].dt.month
            self.data['year_month'] = self.data['acq_date'].dt.to_period('M')
            
            return {"status": "success", "message": "Data loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def filter_data_by_period(self, year=None, month=None):
        """Filter data by specific year and/or month"""
        if self.data is None:
            return None
            
        filtered_data = self.data.copy()
        
        if year is not None:
            filtered_data = filtered_data[filtered_data['year'] == year]
        
        if month is not None:
            filtered_data = filtered_data[filtered_data['month'] == month]
            
        return filtered_data

    def train_polynomial_model(self, filtered_data, degree=4):
        """Train polynomial regression model on filtered data"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            
            # Aggregate data by month
            fires_per_month = filtered_data.groupby('year_month').size().reset_index(name='fire_count')
            fires_per_month['year_month_str'] = fires_per_month['year_month'].astype(str)
            
            # Convert to timestamp for regression
            fires_per_month['timestamp'] = pd.to_datetime(fires_per_month['year_month_str']).astype(int) / 10**9
            
            # Prepare features and target
            X = fires_per_month['timestamp'].values.reshape(-1, 1)
            y = fires_per_month['fire_count'].values
            
            # Train polynomial model
            poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            poly_model.fit(X, y)
            
            # Generate predictions
            fires_per_month['poly_pred'] = poly_model.predict(X)
            
            return {
                "model": poly_model,
                "data": fires_per_month,
                "X": X,
                "y": y
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_polynomial_analysis(self, year=None, month=None, degree=4):
        """Generate polynomial regression analysis"""
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Train model
            model_result = self.train_polynomial_model(filtered_data, degree)
            if "error" in model_result:
                return model_result
                
            poly_model = model_result["model"]
            fires_per_month = model_result["data"]
            X = model_result["X"]
            y = model_result["y"]
            
            # Generate future predictions
            last_timestamp = X.max()
            future_periods = 12  # Predict 12 months ahead
            future_timestamps = np.linspace(last_timestamp, 
                                          last_timestamp + (365.25 * 24 * 3600), 
                                          future_periods).reshape(-1, 1)
            future_predictions = poly_model.predict(future_timestamps)
            
            # Create future dates
            last_date = pd.to_datetime(fires_per_month['year_month_str'].max())
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=future_periods, freq='M')
            
            # Generate plots
            plots = {}
            
            # Main trend plot
            fig1, ax1 = plt.subplots(figsize=(14, 8))
            ax1.scatter(range(len(fires_per_month)), fires_per_month['fire_count'], 
                       label='Actual Fire Count', alpha=0.7, s=60, color='red')
            ax1.plot(range(len(fires_per_month)), fires_per_month['poly_pred'], 
                    label=f'Polynomial Trend (Degree {degree})', linewidth=3, color='green')
            
            # Add future predictions
            future_x = range(len(fires_per_month), len(fires_per_month) + future_periods)
            ax1.plot(future_x, future_predictions, 
                    label='Future Predictions', linewidth=3, color='blue', linestyle='--')
            
            ax1.set_title(f"Polynomial Regression Analysis - {year or 'All Years'} {month or 'All Months'}")
            ax1.set_xlabel("Time Period")
            ax1.set_ylabel("Fire Count")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis labels
            all_labels = fires_per_month['year_month_str'].tolist() + [d.strftime('%Y-%m') for d in future_dates]
            ax1.set_xticks(range(0, len(all_labels), max(1, len(all_labels)//10)))
            ax1.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), max(1, len(all_labels)//10))], 
                              rotation=45, ha='right')
            
            plots['trend_analysis'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Residuals plot
            residuals = fires_per_month['fire_count'] - fires_per_month['poly_pred']
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.scatter(fires_per_month['poly_pred'], residuals, alpha=0.7, color='purple')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title(f"Residuals Plot - Polynomial Degree {degree}")
            ax2.set_xlabel("Predicted Fire Count")
            ax2.set_ylabel("Residuals")
            ax2.grid(True, alpha=0.3)
            plots['residuals'] = self._fig_to_base64(fig2)
            plt.close(fig2)
            
            # Feature importance/coefficients visualization
            if hasattr(poly_model.named_steps['linearregression'], 'coef_'):
                coefficients = poly_model.named_steps['linearregression'].coef_
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.bar(range(len(coefficients)), coefficients, color='orange', alpha=0.7)
                ax3.set_title(f"Polynomial Coefficients (Degree {degree})")
                ax3.set_xlabel("Polynomial Term")
                ax3.set_ylabel("Coefficient Value")
                ax3.grid(True, alpha=0.3)
                plots['coefficients'] = self._fig_to_base64(fig3)
                plt.close(fig3)
            
            # Calculate model metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y, fires_per_month['poly_pred'])
            r2 = r2_score(y, fires_per_month['poly_pred'])
            mae = mean_absolute_error(y, fires_per_month['poly_pred'])
            
            # Prepare CSV data
            historical_data = fires_per_month[['year_month_str', 'fire_count', 'poly_pred']].copy()
            historical_data.columns = ['period', 'actual_count', 'predicted_count']
            
            future_data = pd.DataFrame({
                'period': [d.strftime('%Y-%m') for d in future_dates],
                'predicted_count': future_predictions,
                'type': 'forecast'
            })
            
            csv_data = {
                'historical': historical_data.to_dict('records'),
                'forecast': future_data.to_dict('records')
            }
            
            return {
                "status": "success",
                "plots": plots,
                "csv_data": csv_data,
                "model_metrics": {
                    "mse": float(mse),
                    "r2_score": float(r2),
                    "mae": float(mae),
                    "polynomial_degree": degree
                },
                "summary": {
                    "total_periods": int(len(fires_per_month)),
                    "avg_fire_count": float(fires_per_month['fire_count'].mean()),
                    "predicted_avg": float(fires_per_month['poly_pred'].mean()),
                    "future_avg_prediction": float(future_predictions.mean()),
                    "trend_direction": "increasing" if future_predictions[-1] > y[-1] else "decreasing"
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def compare_polynomial_degrees(self, year=None, month=None, degrees=[2, 3, 4, 5]):
        """Compare different polynomial degrees"""
        try:
            filtered_data = self.filter_data_by_period(year, month)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            from sklearn.metrics import mean_squared_error, r2_score
            
            comparison_results = []
            
            for degree in degrees:
                model_result = self.train_polynomial_model(filtered_data, degree)
                if "error" not in model_result:
                    fires_per_month = model_result["data"]
                    y_true = model_result["y"]
                    y_pred = fires_per_month['poly_pred']
                    
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    comparison_results.append({
                        "degree": degree,
                        "mse": float(mse),
                        "r2_score": float(r2),
                        "avg_prediction": float(y_pred.mean())
                    })
            
            # Create comparison plot
            if comparison_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                degrees_list = [r["degree"] for r in comparison_results]
                mse_list = [r["mse"] for r in comparison_results]
                r2_list = [r["r2_score"] for r in comparison_results]
                
                ax1.bar(degrees_list, mse_list, color='red', alpha=0.7)
                ax1.set_title("Mean Squared Error by Polynomial Degree")
                ax1.set_xlabel("Polynomial Degree")
                ax1.set_ylabel("MSE")
                ax1.grid(True, alpha=0.3)
                
                ax2.bar(degrees_list, r2_list, color='blue', alpha=0.7)
                ax2.set_title("R² Score by Polynomial Degree")
                ax2.set_xlabel("Polynomial Degree")
                ax2.set_ylabel("R² Score")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                comparison_plot = self._fig_to_base64(fig)
                plt.close(fig)
                
                # Find best degree
                best_degree = min(comparison_results, key=lambda x: x["mse"])
                
                return {
                    "status": "success",
                    "comparison_plot": comparison_plot,
                    "comparison_data": comparison_results,
                    "best_degree": best_degree,
                    "summary": {
                        "degrees_tested": degrees,
                        "best_performing_degree": best_degree["degree"],
                        "best_mse": best_degree["mse"],
                        "best_r2": best_degree["r2_score"]
                    }
                }
            else:
                return {"error": "No valid models could be trained"}
                
        except Exception as e:
            return {"error": str(e)}

    def run_comprehensive_polynomial_analysis(self, year=None, month=None, degree=4):
        """Run comprehensive polynomial analysis"""
        try:
            if self.data is None:
                load_result = self.load_data()
                if load_result["status"] == "error":
                    return load_result

            results = {
                "status": "success",
                "model_type": "polynomial_regression",
                "period": f"{year or 'All Years'} - {month or 'All Months'}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Run main polynomial analysis
            poly_result = self.generate_polynomial_analysis(year, month, degree)
            if poly_result.get("status") == "success":
                results["polynomial_analysis"] = poly_result
            else:
                results["polynomial_analysis"] = {"error": poly_result.get("error")}
            
            # Run degree comparison
            comparison_result = self.compare_polynomial_degrees(year, month)
            if comparison_result.get("status") == "success":
                results["degree_comparison"] = comparison_result
            else:
                results["degree_comparison"] = {"error": comparison_result.get("error")}
            
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{image_base64}"