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
import multer from "multer";
import csv from "csv-parser";
import fs from "fs";
import axios from "axios";
import { GoogleGenerativeAI } from "@google/generative-ai";

import analyticsRoutes from "./routes/analytics.js";



dotenv.config()
import { fileURLToPath } from "url"
const app = express()

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PORT = process.env.PORT || 3000;

app.set('view engine', "ejs")
app.set("views", path.join(path.resolve(), "views"));
app.set("views",path.join(__dirname,"views"));
app.use(bodyParser.urlencoded({ extended: true }));

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


// Multer setup for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});
const upload = multer({ storage });



// GET route to render upload page
// app.get("/aiinsights", (req, res) => {
//   res.render("aiinsights", { analysis: null, aiInsights: null });
// });
// POST route to handle CSV upload
// app.post("/aiinsights", upload.single("csvFile"), async (req, res) => {
//   try {
//     const filePath = req.file.path;
//     const csvData = [];

//     // Parse CSV
//     fs.createReadStream(filePath)
//       .pipe(csv())
//       .on("data", (row) => csvData.push(row))
//       .on("end", async () => {
//         // Prepare prompt for Gemini API
//         const prompt = `
// You are an environmental data analyst. Analyze the following CSV data and provide actionable insights to reduce carbon emissions. The CSV data is:
// ${JSON.stringify(csvData)}

// Provide:
// 1. Summary of top contributors to carbon emissions.
// 2. Practical recommendations to reduce emissions.
// 3. Bullet points and numerical estimates if possible.
// Use friendly and simple language.
// `;

//         // Gemini API call
//         const response = await axios.post(
//           "https://api.gemini.com/v1/completions", // Replace with actual Gemini endpoint
//           {
//             model: "gemini-1.5", // Replace with your Gemini model if needed
//             prompt: prompt,
//             max_tokens: 500,
//           },
//           {
//             headers: {
//               Authorization: `Bearer ${process.env.GEMINI_API_KEY}`, // Replace with your key
//               "Content-Type": "application/json",
//             },
//           }
//         );

//         const insights = response.data.choices[0].text || "No insights generated.";

//         // Render page with insights
//         res.render("aiinsights", { insights });
//       });
//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Error processing CSV or calling AI API.");
//   }
// });

// POST route -> process form
// app.post("/aiinsights", async (req, res) => {
//   const { totalCars, carType, personsPerCar, totalPersons } = req.body;

//   let analysis = [];

//   // Hardcoded analysis logic
//   if (carType === "petrol" || carType === "diesel") {
//     analysis.push(`Using a ${carType} car increases carbon emissions significantly.`);
//   } else if (carType === "electric") {
//     analysis.push("Great choice! Electric vehicles have much lower carbon emissions.");
//   }

//   if (personsPerCar <= 4 && totalPersons <= 4) {
//     analysis.push("This is carpooling, which helps reduce emissions.");
//   }

//   if (personsPerCar == 1) {
//     analysis.push("Since you’re traveling alone, consider using a bus or shuttle instead.");
//   }

//   // Send data to AI API (replace with your real key)
//   let aiInsights = "AI insights not available.";
//   try {
//     const response = await axios.post(
//       "https://api.openai.com/v1/chat/completions",
//       {
//         model: "gpt-3.5-turbo",
//         messages: [
//           {
//             role: "system",
//             content: "You are an environmental expert giving carbon reduction advice.",
//           },
//           {
//             role: "user",
//             content: `Analyze this travel scenario:
//             Cars: ${totalCars}, Type: ${carType}, Persons per car: ${personsPerCar}, Total persons: ${totalPersons}.
//             Give suggestions to reduce carbon emissions.`,
//           },
//         ],
//       },
//       {
//         headers: {
//   "Authorization": `Bearer ${GEMINI_API_KEY}`,
//   "Content-Type": "application/json"
// }

//       }
//     );
//     aiInsights = response.data.choices[0].message.content;
//   } catch (err) {
//     console.error(err.message);
//   }

//   res.render("aiinsights", { analysis, aiInsights });
// });

// Ensure 'uploads' folder exists
// if (!fs.existsSync("uploads")) fs.mkdirSync("uploads");



// Initialize Gemini
// const genAI = new GoogleGenerativeAI(`${process.env.GEMINI_API_KEY}`);

// GET route -> form
// app.get("/aiinsights", (req, res) => {
//   res.render("aiinsights", { analysis: null, aiInsights: null });
// });

// // POST route -> process form
// app.post("/aiinsights", async (req, res) => {
//   const { totalCars, carType, personsPerCar, totalPersons } = req.body;

//   let analysis = [];

//   // Hardcoded logic
//   if (carType === "petrol" || carType === "diesel") {
//     analysis.push(`Using a ${carType} car increases carbon emissions significantly.`);
//   } else if (carType === "electric") {
//     analysis.push("Great choice! Electric vehicles have much lower carbon emissions.");
//   }

//   if (personsPerCar <= 4 && totalPersons <= 4) {
//     analysis.push("This is carpooling, which helps reduce emissions.");
//   }

//   if (personsPerCar == 1) {
//     analysis.push("Since you’re traveling alone, consider using a bus or shuttle instead.");
//   }

//   // Gemini AI Call
//   let aiInsights = "AI insights not available.";
//   try {
//     const model = genAI.getGenerativeModel({ model: "models/gemini-1.5-pro-latest" });

//     const prompt = `Analyze this travel scenario:
//       Cars: ${totalCars}, 
//       Type: ${carType}, 
//       Persons per car: ${personsPerCar}, 
//       Total persons: ${totalPersons}.
//       Provide smart suggestions to reduce carbon emissions in simple points.`;

//     const result = await model.generateContent(prompt);
//     aiInsights = result.response.text();
//   } catch (err) {
//     console.error("Gemini API Error:", err.message);
//   }

//   res.render("aiinsights", { analysis, aiInsights });
// });


// // ✅ Initialize Gemini with your API Key
// const gemini_api_key = "AIzaSyAzxvAGURWwHS8npAx1hh_gM29fT0fusJ4";
// const genAI = new GoogleGenerativeAI(gemini_api_key);

// // GET route -> form
// app.get("/aiinsights", (req, res) => {
//   res.render("aiinsights", { analysis: null, aiInsights: null });
// });

// // POST route -> process form
// app.post("/aiinsights", async (req, res) => {
//   const { totalCars, carType, personsPerCar, totalPersons } = req.body;

//   let analysis = [];

//   // 🔹 Hardcoded logic
//   if (carType === "petrol" || carType === "diesel") {
//     analysis.push(`Using a ${carType} car increases carbon emissions significantly.`);
//   } else if (carType === "electric") {
//     analysis.push("Great choice! Electric vehicles have much lower carbon emissions.");
//   }

//   if (personsPerCar <= 4 && totalPersons <= 4) {
//     analysis.push("This is carpooling, which helps reduce emissions.");
//   }

//   if (personsPerCar == 1) {
//     analysis.push("Since you’re traveling alone, consider using a bus or shuttle instead.");
//   }

//   // 🔹 Gemini AI Call
//   let aiInsights = "AI insights not available.";
//   try {
//     // Use latest Gemini model (choose flash for speed, pro for accuracy)
//     const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

//     const prompt = `You are an environmental expert. Analyze this travel scenario:
//     - Cars: ${totalCars} 
//     - Car Type: ${carType} 
//     - Persons per car: ${personsPerCar} 
//     - Total persons: ${totalPersons} 

//     Provide practical and simple suggestions to reduce carbon emissions. 
//     Answer in short bullet points.`;

//     const result = await model.generateContent(prompt);

//     // Extract text response
//     aiInsights = result.response.text();
//   } catch (err) {
//     console.error("Gemini API Error:", err.message);
//   }

//   res.render("aiinsights", { analysis, aiInsights });
// });



// ✅ Gemini Initialization
const gemini_api_key = "AIzaSyAzxvAGURWwHS8npAx1hh_gM29fT0fusJ4";
const genAI = new GoogleGenerativeAI(gemini_api_key);

app.get("/aiinsights", (req, res) => {
  res.render("aiinsights", { analysis: null, aiInsights: null });
});

app.post("/aiinsights", async (req, res) => {
  const { totalCars, carType, personsPerCar, totalPersons } = req.body;

  let analysis = [];

  // Hardcoded logic
  if (carType === "petrol" || carType === "diesel") {
    analysis.push(`Using a ${carType} car increases carbon emissions significantly.`);
  } else if (carType === "electric") {
    analysis.push("Great choice! Electric vehicles have much lower carbon emissions.");
  }

  if (personsPerCar <= 4 && totalPersons <= 4) {
    analysis.push("This is carpooling, which helps reduce emissions.");
  }

  if (personsPerCar == 1) {
    analysis.push("Since you’re traveling alone, consider using a bus or shuttle instead.");
  }

  // 🔹 Gemini AI Call
  let aiInsights = "AI insights not available.";
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });


  const prompt = `You are an environmental expert. Analyze this travel scenario:
  - Cars: ${totalCars}
  - Car Type: ${carType}
  - Persons per car: ${personsPerCar}
  - Total persons: ${totalPersons}

  Provide practical suggestions to reduce carbon emissions. 
  Answer in 3–5 short bullet points.`;

  const result = await model.generateContent(prompt);

  // 👇 Add this here to debug full response
  console.log("Gemini raw result:", JSON.stringify(result, null, 2));

  // ✅ Extract AI text
  aiInsights = await result.response.text();

  } catch (err) {
    console.error("Gemini API Error:", err);
  aiInsights = `
    - Use public transport for solo trips
    - Prefer carpooling when possible
    - Switch to EV or hybrid vehicles
    - Walk or cycle for short distances
  `;
  }

  res.render("aiinsights", { analysis, aiInsights });
});



app.listen(PORT, (req,res) => {
    console.log(`Server is connected to port ${PORT}`)
})







