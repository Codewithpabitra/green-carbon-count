import { Router } from "express";
import { authenticate, isOrg, isStudent } from "../middleware/auth.js";

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

router.get("/community", authenticate, (req, res) => {
  res.render("community")
});

// ORG(Admin Routes)
router.get("/aiinsights", authenticate, isOrg, (req, res) => {
  res.render("aiinsights")
});

router.get("/analytics", authenticate, isOrg, (req, res) => {
  res.render("analytics")
});

export default router;

