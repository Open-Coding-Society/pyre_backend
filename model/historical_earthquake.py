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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import io
import base64
import json
from sklearn.manifold import TSNE
import hdbscan
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
import os

class EarthquakeDataAnalysisAdvancedRegressionModel:
    _instance = None

    @staticmethod
    def get_instance():
        if EarthquakeDataAnalysisAdvancedRegressionModel._instance is None:
            EarthquakeDataAnalysisAdvancedRegressionModel()
        return EarthquakeDataAnalysisAdvancedRegressionModel._instance

    def __init__(self):
        if EarthquakeDataAnalysisAdvancedRegressionModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EarthquakeDataAnalysisAdvancedRegressionModel._instance = self
            self.data = None
            self.prophet_model = None
            self.clustering_model = None
            self.scaler = StandardScaler()
            
    def load_data(self, file_path="earthquakes.csv"):
        """Load and preprocess the earthquake data"""
        try:
            self.data = pd.read_csv(file_path)
            
            # Convert time to datetime
            self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')
            self.data = self.data.dropna(subset=['time'])
            
            # Filter data to reasonable range (last 10 years)
            current_year = datetime.now().year
            self.data = self.data[(self.data['time'].dt.year >= current_year - 10) & 
                                 (self.data['time'].dt.year <= current_year)]
            
            # Clean numeric columns
            self.data['mag'] = pd.to_numeric(self.data['mag'], errors='coerce')
            self.data['depth'] = pd.to_numeric(self.data['depth'], errors='coerce')
            self.data = self.data.dropna(subset=['latitude', 'longitude', 'mag'])
            
            # Filter for actual earthquakes
            if 'type' in self.data.columns:
                self.data = self.data[self.data['type'].str.contains('earthquake', case=False, na=False)]
            
            # Add additional time features
            self.data['year'] = self.data['time'].dt.year
            self.data['month'] = self.data['time'].dt.month
            self.data['day_of_year'] = self.data['time'].dt.dayofyear
            self.data['year_month'] = self.data['time'].dt.to_period('M')
            self.data['acq_date'] = self.data['time'].dt.date  # For consistency with fire model
            
            return {"status": "success", "message": "Earthquake data loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def filter_data_by_period(self, year=None, month=None, magnitude_threshold=None):
        """Filter data by specific year, month, and/or magnitude"""
        if self.data is None:
            return None
            
        filtered_data = self.data.copy()
        
        if year is not None:
            filtered_data = filtered_data[filtered_data['year'] == year]
        
        if month is not None:
            filtered_data = filtered_data[filtered_data['month'] == month]
            
        if magnitude_threshold is not None:
            filtered_data = filtered_data[filtered_data['mag'] >= magnitude_threshold]
            
        return filtered_data

    def generate_time_series_analysis(self, year=None, month=None, magnitude_threshold=None):
        try:
            filtered_data = self.filter_data_by_period(year, month, magnitude_threshold)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Prepare data for Prophet
            if month is not None:
                # Daily aggregation for specific month
                daily_earthquakes = filtered_data.groupby(filtered_data['time'].dt.date).agg({
                    'mag': ['count', 'mean', 'max'],
                    'depth': 'mean'
                }).reset_index()
                daily_earthquakes.columns = ['ds', 'earthquake_count', 'avg_magnitude', 'max_magnitude', 'avg_depth']
                daily_earthquakes['ds'] = pd.to_datetime(daily_earthquakes['ds'])
                daily_earthquakes['y'] = daily_earthquakes['earthquake_count']
                freq = 'D'
                periods = 30  # Forecast 30 days
            else:
                # Monthly aggregation
                monthly_earthquakes = filtered_data.groupby('year_month').agg({
                    'mag': ['count', 'mean', 'max'],
                    'depth': 'mean'
                }).reset_index()
                monthly_earthquakes.columns = ['year_month', 'earthquake_count', 'avg_magnitude', 'max_magnitude', 'avg_depth']
                monthly_earthquakes['ds'] = pd.to_datetime(monthly_earthquakes['year_month'].astype(str))
                monthly_earthquakes['y'] = monthly_earthquakes['earthquake_count']
                daily_earthquakes = monthly_earthquakes
                freq = 'M'
                periods = 12  # Forecast 12 months

            # Train Prophet model
            model = Prophet(yearly_seasonality=True, daily_seasonality=(month is not None))
            model.fit(daily_earthquakes[['ds', 'y']])
            
            # Generate forecast
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            
            # Generate plots
            plots = {}
            
            # Forecast plot
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            model.plot(forecast, ax=ax1)
            title_suffix = f"Mag {magnitude_threshold}+" if magnitude_threshold else "All Magnitudes"
            ax1.set_title(f"Earthquake Count Forecast - {year or 'All Years'} {month or 'All Months'} ({title_suffix})")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Earthquake Count")
            plots['forecast'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Components plot
            fig2 = model.plot_components(forecast)
            fig2.suptitle(f"Forecast Components - {year or 'All Years'} {month or 'All Months'}")
            plots['components'] = self._fig_to_base64(fig2)
            plt.close(fig2)
            
            # Magnitude vs Frequency Analysis
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Magnitude distribution
            filtered_data['mag'].hist(bins=30, ax=ax3a, alpha=0.7, color='red')
            ax3a.set_title(f"Magnitude Distribution")
            ax3a.set_xlabel("Magnitude")
            ax3a.set_ylabel("Frequency")
            ax3a.grid(True, alpha=0.3)
            
            # Depth vs Magnitude
            ax3b.scatter(filtered_data['mag'], filtered_data['depth'], alpha=0.5, color='blue')
            ax3b.set_title("Depth vs Magnitude")
            ax3b.set_xlabel("Magnitude")
            ax3b.set_ylabel("Depth (km)")
            ax3b.grid(True, alpha=0.3)
            
            plots['magnitude_analysis'] = self._fig_to_base64(fig3)
            plt.close(fig3)
            
            # Generate CSV data
            csv_data = {
                'historical': daily_earthquakes.to_dict('records'),
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                'magnitude_stats': {
                    'avg_magnitude': float(filtered_data['mag'].mean()),
                    'max_magnitude': float(filtered_data['mag'].max()),
                    'min_magnitude': float(filtered_data['mag'].min()),
                    'avg_depth': float(filtered_data['depth'].mean()) if 'depth' in filtered_data.columns else None
                }
            }
            
            return {
                "status": "success",
                "plots": plots,
                "csv_data": csv_data,
                "summary": {
                    "total_earthquakes": int(filtered_data.shape[0]),
                    "avg_daily_earthquakes": float(daily_earthquakes['y'].mean()),
                    "max_daily_earthquakes": int(daily_earthquakes['y'].max()),
                    "forecast_avg": float(forecast['yhat'].tail(periods).mean()),
                    "magnitude_stats": csv_data['magnitude_stats']
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_spatial_analysis(self, year=None, month=None, magnitude_threshold=None):
        """Generate spatial clustering analysis for earthquakes"""
        try:
            filtered_data = self.filter_data_by_period(year, month, magnitude_threshold)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Prepare spatial data
            spatial_data = filtered_data[['latitude', 'longitude', 'mag', 'depth']].dropna()
            if len(spatial_data) < 10:
                return {"error": "Insufficient spatial data for clustering"}

            # Perform clustering on location
            n_clusters = min(8, len(spatial_data) // 10)  # Adaptive cluster count
            location_features = spatial_data[['latitude', 'longitude']].values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(location_features)
            
            # Create spatial analysis plots
            plots = {}
            
            # Earthquake epicenter map with magnitude
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            scatter = ax1.scatter(spatial_data['longitude'], spatial_data['latitude'], 
                                c=spatial_data['mag'], cmap='Reds', alpha=0.7, 
                                s=spatial_data['mag']*10, edgecolors='black', linewidth=0.5)
            ax1.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
                       c='blue', marker='x', s=200, linewidths=3, label='Cluster Centers')
            
            title_suffix = f"Mag {magnitude_threshold}+" if magnitude_threshold else "All Magnitudes"
            ax1.set_title(f"Earthquake Epicenters - {year or 'All Years'} {month or 'All Months'} ({title_suffix})")
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")
            ax1.legend()
            
            # Add colorbar for magnitude
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Magnitude')
            plots['epicenter_map'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Cluster analysis with magnitude and depth
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Magnitude by cluster
            spatial_data_with_clusters = spatial_data.copy()
            spatial_data_with_clusters['cluster'] = clusters
            
            cluster_mag_data = []
            for i in range(n_clusters):
                cluster_mags = spatial_data_with_clusters[spatial_data_with_clusters['cluster'] == i]['mag']
                cluster_mag_data.extend(cluster_mags.tolist())
                ax2a.scatter([i] * len(cluster_mags), cluster_mags, alpha=0.6, s=30)
            
            ax2a.set_title("Magnitude Distribution by Cluster")
            ax2a.set_xlabel("Cluster")
            ax2a.set_ylabel("Magnitude")
            ax2a.grid(True, alpha=0.3)
            
            # Depth by cluster
            for i in range(n_clusters):
                cluster_depths = spatial_data_with_clusters[spatial_data_with_clusters['cluster'] == i]['depth']
                ax2b.scatter([i] * len(cluster_depths), cluster_depths, alpha=0.6, s=30)
            
            ax2b.set_title("Depth Distribution by Cluster")
            ax2b.set_xlabel("Cluster")
            ax2b.set_ylabel("Depth (km)")
            ax2b.grid(True, alpha=0.3)
            
            plots['cluster_analysis'] = self._fig_to_base64(fig2)
            plt.close(fig2)
            
            # Cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_earthquakes = spatial_data[cluster_mask]
                cluster_stats.append({
                    'cluster_id': int(i),
                    'earthquake_count': int(cluster_mask.sum()),
                    'center_lat': float(kmeans.cluster_centers_[i, 0]),
                    'center_lon': float(kmeans.cluster_centers_[i, 1]),
                    'avg_magnitude': float(cluster_earthquakes['mag'].mean()),
                    'max_magnitude': float(cluster_earthquakes['mag'].max()),
                    'avg_depth': float(cluster_earthquakes['depth'].mean()),
                    'avg_lat': float(cluster_earthquakes['latitude'].mean()),
                    'avg_lon': float(cluster_earthquakes['longitude'].mean())
                })
            
            return {
                "status": "success",
                "plots": plots,
                "cluster_data": cluster_stats,
                "summary": {
                    "total_earthquakes": int(len(spatial_data)),
                    "n_clusters": n_clusters,
                    "highest_magnitude_cluster": max(cluster_stats, key=lambda x: x['max_magnitude']),
                    "most_active_cluster": max(cluster_stats, key=lambda x: x['earthquake_count'])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_statistical_summary(self, year=None, month=None, magnitude_threshold=None):
        """Generate comprehensive statistical analysis for earthquakes"""
        try:
            filtered_data = self.filter_data_by_period(year, month, magnitude_threshold)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Generate summary statistics
            plots = {}
            
            # Monthly distribution
            monthly_counts = filtered_data.groupby('month').size()
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            monthly_counts.plot(kind='bar', ax=ax1, color='darkred', alpha=0.7)
            title_suffix = f"Mag {magnitude_threshold}+" if magnitude_threshold else "All Magnitudes"
            ax1.set_title(f"Earthquake Distribution by Month - {year or 'All Years'} ({title_suffix})")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Earthquake Count")
            ax1.grid(True, alpha=0.3)
            plots['monthly_distribution'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Yearly trend (if not filtered by specific year)
            if year is None:
                yearly_counts = filtered_data.groupby('year').size()
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                yearly_counts.plot(kind='line', ax=ax2, marker='o', linewidth=2, markersize=8, color='darkblue')
                ax2.set_title(f"Earthquake Count Trend by Year ({title_suffix})")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Earthquake Count")
                ax2.grid(True, alpha=0.3)
                plots['yearly_trend'] = self._fig_to_base64(fig2)
                plt.close(fig2)
            
            # Magnitude-Frequency relationship (Gutenberg-Richter Law)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            magnitude_bins = np.arange(filtered_data['mag'].min(), filtered_data['mag'].max() + 0.5, 0.5)
            mag_counts = []
            mag_centers = []
            
            for i in range(len(magnitude_bins) - 1):
                count = len(filtered_data[(filtered_data['mag'] >= magnitude_bins[i]) & 
                                        (filtered_data['mag'] < magnitude_bins[i+1])])
                mag_counts.append(count)
                mag_centers.append((magnitude_bins[i] + magnitude_bins[i+1]) / 2)
            
            # Plot on log scale
            ax3.semilogy(mag_centers, mag_counts, 'ro-', markersize=6, linewidth=2)
            ax3.set_title("Magnitude-Frequency Distribution (Gutenberg-Richter)")
            ax3.set_xlabel("Magnitude")
            ax3.set_ylabel("Frequency (log scale)")
            ax3.grid(True, alpha=0.3)
            plots['magnitude_frequency'] = self._fig_to_base64(fig3)
            plt.close(fig3)
            
            # Generate summary statistics
            summary_stats = {
                "total_earthquakes": int(len(filtered_data)),
                "period": f"{year or 'All Years'} - {month or 'All Months'}",
                "magnitude_filter": magnitude_threshold,
                "date_range": {
                    "start": filtered_data['time'].min().strftime('%Y-%m-%d'),
                    "end": filtered_data['time'].max().strftime('%Y-%m-%d')
                },
                "monthly_breakdown": monthly_counts.to_dict(),
                "peak_month": int(monthly_counts.idxmax()),
                "avg_earthquakes_per_month": float(monthly_counts.mean()),
                "magnitude_stats": {
                    "mean": float(filtered_data['mag'].mean()),
                    "median": float(filtered_data['mag'].median()),
                    "std": float(filtered_data['mag'].std()),
                    "min": float(filtered_data['mag'].min()),
                    "max": float(filtered_data['mag'].max())
                },
                "depth_stats": {
                    "mean": float(filtered_data['depth'].mean()),
                    "median": float(filtered_data['depth'].median()),
                    "std": float(filtered_data['depth'].std()),
                    "min": float(filtered_data['depth'].min()),
                    "max": float(filtered_data['depth'].max())
                } if 'depth' in filtered_data.columns else None
            }
            
            if year is None:
                summary_stats["yearly_breakdown"] = filtered_data.groupby('year').size().to_dict()
            
            return {
                "status": "success",
                "plots": plots,
                "summary": summary_stats,
                "csv_data": filtered_data.groupby(['year', 'month']).agg({
                    'mag': ['count', 'mean', 'max'],
                    'depth': 'mean'
                }).reset_index().to_dict('records')
            }
            
        except Exception as e:
            return {"error": str(e)}

    def run_comprehensive_analysis(self, year=None, month=None, magnitude_threshold=None):
        """Run all analyses and return comprehensive results"""
        try:
            if self.data is None:
                load_result = self.load_data()
                if load_result["status"] == "error":
                    return load_result

            results = {
                "status": "success",
                "period": f"{year or 'All Years'} - {month or 'All Months'}",
                "magnitude_filter": magnitude_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run time series analysis
            ts_result = self.generate_time_series_analysis(year, month, magnitude_threshold)
            if ts_result.get("status") == "success":
                results["time_series"] = ts_result
            else:
                results["time_series"] = {"error": ts_result.get("error")}
            
            # Run spatial analysis
            spatial_result = self.generate_spatial_analysis(year, month, magnitude_threshold)
            if spatial_result.get("status") == "success":
                results["spatial"] = spatial_result
            else:
                results["spatial"] = {"error": spatial_result.get("error")}
            
            # Run statistical summary
            stats_result = self.generate_statistical_summary(year, month, magnitude_threshold)
            if stats_result.get("status") == "success":
                results["statistics"] = stats_result
            else:
                results["statistics"] = {"error": stats_result.get("error")}
            
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

class EarthquakeDataPolynomialRegressionModel:
    _instance = None

    @staticmethod
    def get_instance():
        if EarthquakeDataPolynomialRegressionModel._instance is None:
            EarthquakeDataPolynomialRegressionModel()
        return EarthquakeDataPolynomialRegressionModel._instance

    def __init__(self):
        if EarthquakeDataPolynomialRegressionModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EarthquakeDataPolynomialRegressionModel._instance = self
            self.data = None
            self.poly_model = None
            self.degree = 4  # Default polynomial degree
            
    def load_data(self, file_path="earthquakes.csv"):
        """Load and preprocess the earthquake data"""
        try:
            self.data = pd.read_csv(file_path)
            self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')
            self.data = self.data.dropna(subset=['time'])
            
            # Clean and filter data
            current_year = datetime.now().year
            self.data = self.data[(self.data['time'].dt.year >= current_year - 10)]
            
            self.data['mag'] = pd.to_numeric(self.data['mag'], errors='coerce')
            self.data = self.data.dropna(subset=['latitude', 'longitude', 'mag'])
            
            # Add time features
            self.data['year'] = self.data['time'].dt.year
            self.data['month'] = self.data['time'].dt.month
            self.data['year_month'] = self.data['time'].dt.to_period('M')
            
            return {"status": "success", "message": "Earthquake data loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def filter_data_by_period(self, year=None, month=None, magnitude_threshold=None):
        """Filter data by specific year, month, and/or magnitude"""
        if self.data is None:
            return None
            
        filtered_data = self.data.copy()
        
        if year is not None:
            filtered_data = filtered_data[filtered_data['year'] == year]
        
        if month is not None:
            filtered_data = filtered_data[filtered_data['month'] == month]
            
        if magnitude_threshold is not None:
            filtered_data = filtered_data[filtered_data['mag'] >= magnitude_threshold]
            
        return filtered_data

    def train_polynomial_model(self, filtered_data, degree=4):
        """Train polynomial regression model on filtered earthquake data"""
        try:
            # Aggregate data by month with multiple metrics
            earthquakes_per_month = filtered_data.groupby('year_month').agg({
                'mag': ['count', 'mean', 'max'],
                'depth': 'mean'
            }).reset_index()
            
            earthquakes_per_month.columns = ['year_month', 'earthquake_count', 'avg_magnitude', 'max_magnitude', 'avg_depth']
            earthquakes_per_month['year_month_str'] = earthquakes_per_month['year_month'].astype(str)
            
            # Convert to timestamp for regression
            earthquakes_per_month['timestamp'] = pd.to_datetime(earthquakes_per_month['year_month_str']).astype(int) / 10**9
            
            # Prepare features and target
            X = earthquakes_per_month['timestamp'].values.reshape(-1, 1)
            y = earthquakes_per_month['earthquake_count'].values
            
            # Train polynomial model
            poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            poly_model.fit(X, y)
            
            # Generate predictions
            earthquakes_per_month['poly_pred'] = poly_model.predict(X)
            
            return {
                "model": poly_model,
                "data": earthquakes_per_month,
                "X": X,
                "y": y
            }
            
        except Exception as e:
            return {"error": str(e)}

    def generate_polynomial_analysis(self, year=None, month=None, magnitude_threshold=None, degree=4):
        """Generate polynomial regression analysis for earthquakes"""
        try:
            filtered_data = self.filter_data_by_period(year, month, magnitude_threshold)
            if filtered_data is None or len(filtered_data) == 0:
                return {"error": "No data available for the specified period"}

            # Train model
            model_result = self.train_polynomial_model(filtered_data, degree)
            if "error" in model_result:
                return model_result
                
            poly_model = model_result["model"]
            earthquakes_per_month = model_result["data"]
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
            last_date = pd.to_datetime(earthquakes_per_month['year_month_str'].max())
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=future_periods, freq='M')
            
            # Generate plots
            plots = {}
            
            # Main trend plot
            fig1, ax1 = plt.subplots(figsize=(14, 8))
            ax1.scatter(range(len(earthquakes_per_month)), earthquakes_per_month['earthquake_count'], 
                       label='Actual Earthquake Count', alpha=0.7, s=60, color='red')
            ax1.plot(range(len(earthquakes_per_month)), earthquakes_per_month['poly_pred'], 
                    label=f'Polynomial Trend (Degree {degree})', linewidth=3, color='green')
            
            # Add future predictions
            future_x = range(len(earthquakes_per_month), len(earthquakes_per_month) + future_periods)
            ax1.plot(future_x, future_predictions, 
                    label='Future Predictions', linewidth=3, color='blue', linestyle='--')
            
            title_suffix = f"Mag {magnitude_threshold}+" if magnitude_threshold else "All Magnitudes"
            ax1.set_title(f"Earthquake Polynomial Regression - {year or 'All Years'} {month or 'All Months'} ({title_suffix})")
            ax1.set_xlabel("Time Period")
            ax1.set_ylabel("Earthquake Count")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis labels
            all_labels = earthquakes_per_month['year_month_str'].tolist() + [d.strftime('%Y-%m') for d in future_dates]
            ax1.set_xticks(range(0, len(all_labels), max(1, len(all_labels)//10)))
            ax1.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), max(1, len(all_labels)//10))], 
                              rotation=45, ha='right')
            
            plots['trend_analysis'] = self._fig_to_base64(fig1)
            plt.close(fig1)
            
            # Calculate model metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y, earthquakes_per_month['poly_pred'])
            r2 = r2_score(y, earthquakes_per_month['poly_pred'])
            mae = mean_absolute_error(y, earthquakes_per_month['poly_pred'])
            
            # Prepare CSV data
            historical_data = earthquakes_per_month[['year_month_str', 'earthquake_count', 'poly_pred', 'avg_magnitude', 'max_magnitude']].copy()
            historical_data.columns = ['period', 'actual_count', 'predicted_count', 'avg_magnitude', 'max_magnitude']
            
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
    "summary": {
        "model_degree": degree,
        "r2_score": r2,
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "total_historical_months": len(earthquakes_per_month),
        "total_forecast_months": future_periods,
        "avg_predicted_next_year": float(np.mean(future_predictions)),
        "peak_predicted_month": str(future_dates[np.argmax(future_predictions)].strftime('%Y-%m'))
    }
}

        except Exception as e:
            return {"status": "error", "message": str(e)}
