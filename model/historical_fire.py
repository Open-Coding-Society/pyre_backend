import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
