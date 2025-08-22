import { Router } from "express";
import { authenticate, isOrg, isStudent } from "../middleware/auth.js";
import Organization from "../models/Organization.js";
import Student from "../models/Student.js";
import bodyParser from "body-parser";
import fetch from "node-fetch"; 

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

router.get("/dashboard", authenticate,  (req, res) => {
  res.render("dashboard")
});

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


router.get("/analytics", authenticate, isOrg, (req, res) => {
  res.render("analytics")
});



export default router;

