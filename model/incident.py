const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = 3000;

// TomTom API Configuration
const TOMTOM_API_KEY = 'RWRdUGyJfGqOLKBh2A1GsfMQT22h1ZnX';
const TOMTOM_BASE_URL = 'https://api.tomtom.com/traffic/services/5/incidentDetails';

// Middleware
app.use(cors());
app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Main incidents endpoint
app.get('/api/incidents', async (req, res) => {
    try {
        const { lat, lon, radius = 10 } = req.query;

        // Validate required parameters
        if (!lat || !lon) {
            return res.status(400).json({
                error: 'Missing required parameters',
                message: 'Both latitude (lat) and longitude (lon) are required'
            });
        }

        // Validate coordinate ranges
        const latitude = parseFloat(lat);
        const longitude = parseFloat(lon);
        const radiusKm = parseInt(radius);

        if (isNaN(latitude) || isNaN(longitude) || isNaN(radiusKm)) {
            return res.status(400).json({
                error: 'Invalid parameters',
                message: 'Latitude, longitude, and radius must be valid numbers'
            });
        }

        if (latitude < -90 || latitude > 90) {
            return res.status(400).json({
                error: 'Invalid latitude',
                message: 'Latitude must be between -90 and 90'
            });
        }

        if (longitude < -180 || longitude > 180) {
            return res.status(400).json({
                error: 'Invalid longitude',
                message: 'Longitude must be between -180 and 180'
            });
        }

        if (radiusKm < 1 || radiusKm > 50) {
            return res.status(400).json({
                error: 'Invalid radius',
                message: 'Radius must be between 1 and 50 kilometers'
            });
        }

        console.log(`Fetching incidents for coordinates: ${latitude}, ${longitude} with radius: ${radiusKm}km`);

        // Calculate bounding box for the given radius
        const boundingBox = calculateBoundingBox(latitude, longitude, radiusKm);

        // TomTom API request
        const tomtomUrl = `${TOMTOM_BASE_URL}`;
        const params = {
            key: TOMTOM_API_KEY,
            bbox: `${boundingBox.minLon},${boundingBox.minLat},${boundingBox.maxLon},${boundingBox.maxLat}`,
            fields: '{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,startTime,endTime,from,to,length,delay,roadNumbers,timeValidity,probabilityOfOccurrence,numberOfReports,lastReportTime,description}}}',
            language: 'en-US',
            categoryFilter: '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14',
            timeValidityFilter: 'present'
        };

        const response = await axios.get(tomtomUrl, { 
            params,
            timeout: 10000 // 10 second timeout
        });

        console.log(`TomTom API response status: ${response.status}`);

        if (!response.data || !response.data.incidents) {
            return res.json({
                incidents: [],
                location: { latitude, longitude, radius: radiusKm },
                message: 'No incidents data available from TomTom API'
            });
        }

        // Process and format incidents
        const incidents = response.data.incidents.map(incident => {
            const props = incident.properties || {};
            const geometry = incident.geometry || {};
            
            // Extract coordinates for location description
            let locationDescription = 'Unknown location';
            if (geometry.coordinates && geometry.coordinates.length > 0) {
                const coords = geometry.coordinates[0];
                if (Array.isArray(coords) && coords.length >= 2) {
                    locationDescription = `${coords[1].toFixed(4)}, ${coords[0].toFixed(4)}`;
                }
            }

            // Map TomTom categories to readable types
            const incidentTypeMap = {
                0: 'Unknown',
                1: 'Accident',
                2: 'Fog',
                3: 'Dangerous Conditions',
                4: 'Rain',
                5: 'Ice',
                6: 'Jam',
                7: 'Lane Closed',
                8: 'Road Closed',
                9: 'Road Works',
                10: 'Wind',
                11: 'Flooding',
                12: 'Detour',
                13: 'Cluster'
            };

            return {
                id: incident.id || `incident_${Date.now()}_${Math.random()}`,
                type: incidentTypeMap[props.iconCategory] || 'Traffic Incident',
                severity: mapDelayToSeverity(props.magnitudeOfDelay),
                location: props.from && props.to ? `From ${props.from} to ${props.to}` : locationDescription,
                description: props.description || generateDescriptionFromProps(props),
                startTime: props.startTime || null,
                endTime: props.endTime || null,
                delay: props.delay || props.magnitudeOfDelay || null,
                length: props.length || null,
                roadNumbers: props.roadNumbers || [],
                coordinates: geometry.coordinates || null,
                lastReportTime: props.lastReportTime || null,
                numberOfReports: props.numberOfReports || null,
                probabilityOfOccurrence: props.probabilityOfOccurrence || null
            };
        });

        // Filter incidents within the specified radius (double-check)
        const filteredIncidents = incidents.filter(incident => {
            if (!incident.coordinates || !incident.coordinates[0]) return true;
            
            const coords = incident.coordinates[0];
            if (!Array.isArray(coords) || coords.length < 2) return true;
            
            const distance = calculateDistance(latitude, longitude, coords[1], coords[0]);
            return distance <= radiusKm;
        });

        console.log(`Found ${filteredIncidents.length} incidents within ${radiusKm}km`);

        res.json({
            incidents: filteredIncidents,
            location: { latitude, longitude, radius: radiusKm },
            totalFound: filteredIncidents.length,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error fetching traffic incidents:', error);
        
        let errorMessage = 'Internal server error';
        let statusCode = 500;

        if (error.code === 'ECONNABORTED') {
            errorMessage = 'Request timeout - TomTom API is taking too long to respond';
            statusCode = 504;
        } else if (error.response) {
            statusCode = error.response.status;
            errorMessage = `TomTom API error: ${error.response.status} - ${error.response.statusText}`;
            
            if (error.response.status === 403) {
                errorMessage = 'Invalid API key or access denied to TomTom API';
            } else if (error.response.status === 429) {
                errorMessage = 'Rate limit exceeded for TomTom API';
            }
        } else if (error.request) {
            errorMessage = 'Unable to connect to TomTom API - network error';
            statusCode = 503;
        }

        res.status(statusCode).json({
            error: 'Failed to fetch traffic incidents',
            message: errorMessage,
            timestamp: new Date().toISOString()
        });
    }
});

// Utility functions
function calculateBoundingBox(lat, lon, radiusKm) {
    const latRadian = lat * Math.PI / 180;
    const degLatKm = 110.54; // km per degree latitude
    const degLonKm = 110.54 * Math.cos(latRadian); // km per degree longitude
    
    const deltaLat = radiusKm / degLatKm;
    const deltaLon = radiusKm / degLonKm;
    
    return {
        minLat: lat - deltaLat,
        maxLat: lat + deltaLat,
        minLon: lon - deltaLon,
        maxLon: lon + deltaLon
    };
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in kilometers
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

function mapDelayToSeverity(magnitudeOfDelay) {
    if (!magnitudeOfDelay) return 'low';
    if (magnitudeOfDelay >= 4) return 'high';
    if (magnitudeOfDelay >= 2) return 'medium';
    return 'low';
}

function generateDescriptionFromProps(props) {
    const parts = [];
    
    if (props.roadNumbers && props.roadNumbers.length > 0) {
        parts.push(`on ${props.roadNumbers.join(', ')}`);
    }
    
    if (props.length) {
        parts.push(`affecting ${props.length}m of road`);
    }
    
    if (props.delay) {
        parts.push(`causing ${props.delay} minute delay`);
    }
    
    return parts.length > 0 ? parts.join(', ') : 'Traffic incident reported';
}

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: 'An unexpected error occurred',
        timestamp: new Date().toISOString()
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not found',
        message: 'The requested endpoint does not exist',
        availableEndpoints: [
            'GET /health',
            'GET /api/incidents?lat={latitude}&lon={longitude}&radius={radius_km}'
        ]
    });
});

// Start server
app.listen(8505, () => {
    console.log(`🚀 Traffic Incident Server running on http://localhost:${8505}`);
    console.log(`📊 Health check available at http://localhost:${8505}/health`);
    console.log(`🚨 Incidents API available at http://localhost:${8505}/api/incidents`);
    console.log(`🔑 Using TomTom API key: ${TOMTOM_API_KEY.substring(0, 8)}...`);
});

module.exports = app;