import express from "express"
import mongoose from "mongoose"
import path from 'path'
import navRouter from "./routes/navigation.routes.js"
import authRouter from "./routes/authentication.routes.js"
import orgRoutes from "./routes/org.js"
import studentRoutes from './routes/student.js'
import protectedRoutes from "./routes/protected.js"
import dotenv from "dotenv"
import cors from "cors";
import cookieParser from "cookie-parser";
import bodyParser from "body-parser";

import analyticsRoutes from "./routes/analytics.js";



dotenv.config()
import { fileURLToPath } from "url"
const app = express()

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PORT = process.env.PORT || 3000;

app.set('view engine', "ejs")
app.set("views",path.join(__dirname,"views"))

app.use(express.urlencoded({extended: true}))
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')));
app.use(cookieParser());
app.use(bodyParser.json());

app.use(cors({
  origin: `http://localhost:${PORT}`, // frontend URL
  credentials: true,
  allowedHeaders: ["Content-Type", "Authorization"]
}));

mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log("✅ MongoDB connected"))
    .catch(err => console.error(err));


// app.use("/nav",navRouter) // now the route will be like this -> "/nav/users"
// app.use("/users",authRouter) // now the route will be like this -> "/users/signup"
app.use("/org", orgRoutes)
app.use("/student",studentRoutes)
app.use("/", protectedRoutes)

// Mount ML analytics route
app.use("/ml/api", analyticsRoutes);

// Route to render analytics page
app.get("/analytics", (req, res) => {
  res.sendFile("frontend_updated.html", { root: "../frontend" });
});

// --- Route for carbon footprint calculation ---

app.post("/calculate-footprint", (req, res) => {
  const { electricity, water, waste, transport, gas, transportMode } = req.body;

  // Emission factors
  const carbonFromElectricity = electricity * 0.82;   // kg CO2 per kWh
  const carbonFromWater = water * 0.34;               // kg CO2 per m³
  const carbonFromWaste = waste * 2.5;                // kg CO2 per kg

  // Transport emission factors by mode
  const transportFactors = {
    Car: 0.21,   // kg CO2 per km
    Bus: 0.11,
    Train: 0.05,
    Bike: 0.0    // assume zero
  };

  const carbonFromTransport = transport * (transportFactors[transportMode] || 0.21);
  const carbonFromGas = gas * 2.0;                    // kg CO2 per m³

  const totalCarbon = (
    carbonFromElectricity +
    carbonFromWater +
    carbonFromWaste +
    carbonFromTransport +
    carbonFromGas
  ).toFixed(2);

  // -------- Dynamic Suggestions --------
  let suggestions = [];

  // Electricity
  if (electricity < 100) {
    suggestions.push("Your electricity usage is low 👏. Keep it up!");
  } else if (electricity <= 300) {
    suggestions.push("Consider using energy-efficient appliances to reduce electricity.");
  } else {
    suggestions.push("Your electricity usage is quite high ⚡. Switch to solar panels or reduce AC usage.");
  }

  // Water
  if (water < 10) {
    suggestions.push("Water usage is minimal 💧, great job!");
  } else if (water <= 30) {
    suggestions.push("Try reusing greywater and installing low-flow taps.");
  } else {
    suggestions.push("High water usage detected 🚿. Fix leaks and avoid overwatering plants.");
  }

  // Waste
  if (waste < 5) {
    suggestions.push("Good waste management ♻️.");
  } else if (waste <= 20) {
    suggestions.push("Consider recycling and composting.");
  } else {
    suggestions.push("Too much waste generated 🗑️. Reduce single-use plastics.");
  }

  // Transport
  if (transport < 50) {
    suggestions.push(`Great! Your ${transportMode} travel is already low-carbon.`);
  } else if (transport <= 200) {
    suggestions.push(`Try reducing ${transportMode} trips or combine errands.`);
  } else {
    suggestions.push(`High transport emissions detected from ${transportMode} 🚗. Use public transport or carpool.`);
  }

  // Gas
  if (gas < 10) {
    suggestions.push("Gas usage is under control ✅.");
  } else if (gas <= 50) {
    suggestions.push("Consider improving insulation to save gas.");
  } else {
    suggestions.push("High gas usage 🏠. Service your heating system and check for leaks.");
  }

  res.json({
    totalCarbon,
    suggestions
  });
});





app.listen(PORT, (req,res) => {
    console.log(`Server is connected to port ${PORT}`)
})







