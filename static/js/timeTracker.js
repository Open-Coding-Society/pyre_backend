// filepath: /static/js/timeTracker.js

// A utility object to handle user time tracking.
const TimeTracker = (() => {
    let userId = null;  // User's ID
    let sessionStartTime = null;  // Start time of the session
    let totalTimeSpent = 0;  // Total session duration in seconds

    const TRACKING_API_URL = '/api/track_time';  // Endpoint to send time data

    // Initialize the tracker with the user ID
    function init(id) {
        if (!id) {
            console.error("User ID is required to track time.");
            return;
        }
        userId = id;
        sessionStartTime = Date.now();
        console.log(`Time tracking started for user: ${userId}`);
    }

    // Calculate the total time spent in the current session
    function calculateTimeSpent() {
        if (!sessionStartTime) return totalTimeSpent;
        const currentTime = Date.now();
        const sessionDuration = Math.floor((currentTime - sessionStartTime) / 1000);  // Seconds
        return totalTimeSpent + sessionDuration;
    }

    // Send tracked time to the server
    async function sendTimeData() {
        if (!userId) {
            console.error("Cannot send time data. User ID is not set.");
            return;
        }

        const totalSessionTime = calculateTimeSpent();
        const payload = {
            user_id: userId,
            time_spent: totalSessionTime
        };

        try {
            const response = await fetch(TRACKING_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                console.log("Time data sent successfully:", payload);
            } else {
                console.error("Failed to send time data:", response.statusText);
            }
        } catch (error) {
            console.error("Error sending time data:", error);
        }
    }

    // Stop tracking the session
    function stopTracking() {
        totalTimeSpent = calculateTimeSpent();
        sessionStartTime = null;
        console.log(`Time tracking stopped for user: ${userId}. Total time: ${totalTimeSpent} seconds.`);
        sendTimeData();
    }

    // Attach event listeners for tracking user activity
    function attachListeners() {
        window.addEventListener('beforeunload', stopTracking);
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'hidden') stopTracking();
        });
    }

    return { init, stopTracking, attachListeners };
})();

// Initialize tracker with the user's ID
TimeTracker.init("user123");
TimeTracker.attachListeners();
