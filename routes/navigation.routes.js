import express from "express"

const router = express.Router()

router.get("/dashboard",(req,res) => {
   res.render("dashboard")
})

router.get("/calculator", (req,res) => {
    res.render("calculator")
})

router.get("/aiinsights", (req,res) => {
    res.render("aiinsights")
})

router.get("/analytics", (req,res) => {
    res.render("analytics")
})

router.get("/community", (req,res) => {
    res.render("community")
})

router.get("/hub", (req,res) => {
    res.render("awarnessHub")
})

export default router