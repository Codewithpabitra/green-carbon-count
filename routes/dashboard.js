import express from "express";
// import SensorData from "../models/SensorData.js"

const router = express.Router();

// Example authenticate middleware


// Dashboard route
router.get("/dashboard", authenticate, async (req, res) => {
  try {
    // Fetch latest 10 sensor records
    const sensorData = await SensorData.find().sort({ timestamp: -1 }).limit(10);
    
    // Pass data to EJS
    res.render("dashboard", { sensorData });
  } catch (err) {
    console.error("Error fetching sensor data:", err);
    res.status(500).send("Server Error");
  }
});

export default router;
