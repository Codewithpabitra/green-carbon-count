import express from "express";
import fetch from "node-fetch";  // install with: npm install node-fetch

const router = express.Router();

// Proxy route to ML service
router.post("/upload", async (req, res) => {
  try {
    const mlRes = await fetch("http://localhost:8000/api/upload", {
      method: "POST",
      body: req,
      headers: req.headers
    });
    const data = await mlRes.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: "ML service unavailable" });
  }
});

router.get("/stats", async (req, res) => {
  try {
    const mlRes = await fetch("http://localhost:8000/api/stats" + (req.url.includes("?") ? req.url : ""));
    const data = await mlRes.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: "ML service unavailable" });
  }
});

// You can add more proxies: /monthly, /daily, /leakage, /predict
export default router;
