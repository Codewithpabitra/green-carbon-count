import { Router } from "express";
import { authenticate, isOrg, isStudent } from "../middleware/auth.js";
import Organization from "../models/Organization.js";
import Student from "../models/Student.js";
import bodyParser from "body-parser";
import fetch from "node-fetch"; 
import multer from "multer";
import fs from "fs"
import Papa from "papaparse";

import SensorData from "../models/SensorData.js"
// import { authenticate } from "./middleware/auth.js";

const router = Router();

// Public (everyone can access)
router.get("/", (req, res) => {
  res.render("home")
});

router.get("/signup",(req,res) => {
  res.render("signup")
})

router.get("/org/org-signup",(req,res) => {
  res.render("org-signup")
})

router.get("/org/org-login",(req,res) => {
  res.render("org-login")
})


router.get("/student/student-signup",(req,res) => {
  res.render("student-signup")
})

router.get("/student/student-login",(req,res) => {
  res.render("student-login")
})






// Student Routes 
// router.get("/dash",(req,res) => {
//   res.render("dashboard")
// })

router.get("/dashboard", authenticate,  async (req, res) => {
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


// Dashboard route
// router.get("/dashboard", authenticate, async (req, res) => {
//   try {
//     // Fetch last 10 sensor data records, newest first
//     const allData = await SensorData.find().sort({ timestamp: -1 }).limit(10);

//     // Render dashboard.ejs and pass the data
//     res.render("dashboard", { sensorData: allData });
//   } catch (err) {
//     console.error("Error fetching sensor data:", err);
//     res.status(500).send("Error loading dashboard");
//   }
// });


 


router.get("/calculator", authenticate, (req, res) => {
  res.render("calculator")
});

router.get("/hub", authenticate,  (req, res) => {
  res.render("awarnessHub")
});

// router.get("/community", authenticate, async (req, res) => {

//   try {
//     // Fetch all organizations
//     const organizations = await Organization.find();

//     // Fetch all students with organization info
//     const students = await Student.find().populate("orgId", "orgName");

//     // Pass to EJS
//     res.render("community", { organizations, students });
//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Server error");
//   }

// });





// ORG(Admin Routes)
// router.get("/aiinsights", authenticate, isOrg, (req, res) => {
//   res.render("aiinsights")
// });


// Existing GET route
router.get("/community", authenticate, async (req, res) => {
  try {
    // Fetch all organizations
    const organizations = await Organization.find();

    // Fetch all students with organization info
    const students = await Student.find().populate("orgId", "orgName");

    // Sort students by greenScore descending for leaderboard
    students.sort((a, b) => b.greenScore - a.greenScore);

    // Pass to EJS
    res.render("community", { organizations, students });
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error");
  }
});

// POST route to add challenge
router.post("/add-challenge/:id", authenticate, async (req, res) => {
  try {
    const studentId = req.params.id;
    const student = await Student.findById(studentId);
    if (!student) return res.status(404).json({ message: "Student not found" });

    student.greenScore += 1;
    await student.save();

    res.json({ greenScore: student.greenScore });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});


// router.get("/analytics", authenticate, isOrg, (req, res) => {
//   res.render("analytics")
// });

const upload = multer({ dest: "/tmp" });

// Temporary storage for insights
let lastInsights = null;
let lastChartData = null;

// Your existing GET route (unchanged)
router.get("/analytics", authenticate, isOrg, (req, res) => {
  res.render("analytics");
});


// Handle CSV upload + Gemini analysis
// router.post("/analytics/upload", authenticate, isOrg, upload.single("csvFile"), async (req, res) => {
//   try {
//     const file = fs.readFileSync(req.file.path, "utf8");
//     fs.unlinkSync(req.file.path); // delete temp file

//     // Parse CSV
//     const parsed = Papa.parse(file, { header: true });
//     const jsonData = parsed.data;

//     // Call Gemini API for insights
//     const response = await fetch(
//       "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=AIzaSyAopQ1rESdRTVZIL_eCcca1YGtBQxiBVFw",
//       {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           contents: [
//             {
//               parts: [
//                 { text: `Analyze this dataset and provide insights:\n${JSON.stringify(jsonData.slice(0, 30))}` }
//               ]
//             }
//           ]
//         }),
//       }
//     );

//     const result = await response.json();
//     const insights =
//       result.candidates?.[0]?.content?.parts?.[0]?.text ||
//       "No insights generated.";

//     // Build chart data (sum of numeric columns)
//     let chartData = { labels: [], values: [] };
//     if (jsonData.length > 0) {
//       const firstRow = jsonData[0];
//       for (let key in firstRow) {
//         if (!isNaN(firstRow[key])) {
//           let sum = jsonData.reduce(
//             (acc, row) => acc + (parseFloat(row[key]) || 0),
//             0
//           );
//           chartData.labels.push(key);
//           chartData.values.push(sum);
//         }
//       }
//     }

//     // Save for frontend fetch
//     lastInsights = insights;
//     lastChartData = chartData;

//     res.redirect("/analytics");
//   } catch (err) {
//     console.error(err);
//     res.status(500).send("Error processing CSV file.");
//   }
// });

router.post("/analytics/upload", authenticate, isOrg, upload.single("csvFile"), async (req, res) => {
  try {
    const file = fs.readFileSync(req.file.path, "utf8");
    fs.unlinkSync(req.file.path); // delete temp file

    // Parse CSV safely
    const parsed = Papa.parse(file, { header: true, skipEmptyLines: true });
    const jsonData = parsed.data.filter(row => Object.keys(row).length > 0);

    if (!jsonData || jsonData.length === 0) {
      lastInsights = "Uploaded CSV is empty or invalid.";
      lastChartData = null;
      return res.redirect("/analytics");
    }

    // Call Gemini API for insights
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: `You are a data analyst. Analyze this CSV data (only first 30 rows shown) and give a short, clear summary with patterns, trends, and anomalies:\n\n${JSON.stringify(
                    jsonData.slice(0, 30),
                    null,
                    2
                  )}`
                }
              ]
            }
          ]
        }),
      }
    );

    const result = await response.json();

    // Try multiple ways to extract Gemini response safely
    let insights = "No insights generated.";
    if (result.candidates && result.candidates.length > 0) {
      const parts = result.candidates[0].content?.parts;
      if (parts && parts.length > 0) {
        insights = parts.map(p => p.text || "").join("\n").trim();
      }
    }

    // Build chart data (sum of numeric columns)
    let chartData = { labels: [], values: [] };
    if (jsonData.length > 0) {
      const firstRow = jsonData[0];
      for (let key in firstRow) {
        if (!isNaN(firstRow[key]) && key.trim() !== "") {
          let sum = jsonData.reduce(
            (acc, row) => acc + (parseFloat(row[key]) || 0),
            0
          );
          chartData.labels.push(key);
          chartData.values.push(sum);
        }
      }
    }

    lastInsights = insights || "No insights returned.";
    lastChartData = chartData;

    res.redirect("/analytics");
  } catch (err) {
    console.error("Analytics Upload Error:", err);
    lastInsights = "⚠️ Error processing CSV or fetching insights.";
    lastChartData = null;
    res.redirect("/analytics");
  }
});


// API endpoint for insights
router.get("/analytics/data", authenticate, isOrg, (req, res) => {
  res.json({ insights: lastInsights, chartData: lastChartData });
});



export default router;

