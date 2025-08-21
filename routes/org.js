import { Router } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import crypto from "crypto";
import dotenv from "dotenv"
import Organization from "../models/Organization.js";

const router = Router();
dotenv.config();

// ----------------- SIGNUP -----------------
router.post("/org-signup", async (req, res) => {
  try {
    const { orgName, email, password } = req.body;

    if (!orgName || !email || !password) {
      return res.status(400).send("All fields required");
    }

    const existing = await Organization.findOne({ email });
    if (existing) return res.status(400).send("Organization already exists");

    // Hash password
    const passwordHash = await bcrypt.hash(password, 12);

    // Generate pass key
    const passKey = crypto.randomBytes(8).toString("hex").toUpperCase();

    // Save org
    const org = await Organization.create({ orgName, email, passwordHash, passKey });

    // Send pass key to org for reference
    res.status(201).send(`Signup successful! Your organization pass key: ${org.passKey}`);
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error");
  }
});

// ----------------- LOGIN -----------------
router.post("/org-login", async (req, res) => {
  try {
    const { email, password, passKey } = req.body;

    const org = await Organization.findOne({ email, passKey });
    if (!org) return res.status(400).send("Invalid credentials");

    const valid = await bcrypt.compare(password, org.passwordHash);
    if (!valid) return res.status(400).send("Invalid credentials");

    // JWT token
    const token = jwt.sign(
      { id: org._id, orgName: org.orgName, role: "org" },
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