import mongoose from "mongoose";

const SensorSchema = new mongoose.Schema({
  temperature: Number,
  humidity: Number,
  air_quality: Number,
  weight: Number,
  voltage: Number,
  current: Number,
  power: Number,
  energy: Number,
  flow_rate: Number,
  timestamp: { type: Date, default: Date.now }
});

const SensorData = mongoose.model("SensorData", SensorSchema);

export default SensorData;
