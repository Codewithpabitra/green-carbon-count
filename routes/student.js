import { Router } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Organization from "../models/Organization.js";
import Student from "../models/Student.js";

const router = Router();

// STUDENT SIGNUP 
router.post("/student-signup", async (req, res) => {
  try {
    const { name, password, passKey } = req.body;
    if (!name || !password || !passKey) {
      return res.status(400).send("All fields required");
    }

    // Find org by passKey
    const org = await Organization.findOne({ passKey });
    if (!org) return res.status(400).send("Invalid organization pass key");

    const passwordHash = await bcrypt.hash(password, 12);

    const student = await Student.create({ orgId: org._id, name, passwordHash });
    res.status(201).send("Student registered successfully!");
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error");
  }
});

// STUDENT LOGIN
router.post("/student-login", async (req, res) => {
  try {
    const { name, password, passKey } = req.body;

    const org = await Organization.findOne({ passKey });
    if (!org) return res.status(400).send("Invalid pass key");

    const student = await Student.findOne({ orgId: org._id, name });
    if (!student) return res.status(400).send("Invalid credentials");

    const valid = await bcrypt.compare(password, student.passwordHash);
    if (!valid) return res.status(400).send("Invalid credentials");

    const token = jwt.sign(
      { id: student._id, name: student.name, orgId: org._id, role: "student" },
      process.env.JWT_SECRET,
      { expiresIn: "1d" }
    );

    res.cookie("token", token, { httpOnly: true }).redirect("/dashboard");
    // res.json({ message: "Login successful", token });
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error");
  }
});

export default router;