const express = require("express")

const router = express.Router()

router.get("/dashboard",(req,res) => {
   res.render("dashboard")
})

router.get("/calculator", (req,res) => {
    res.render("calculator")
})

router.get("/aiinsights", (req,res) => {
    res.send("hello from ai insights")
})

router.get("/analytics", (req,res) => {
    res.send("hello from analytics")
})

router.get("/community", (req,res) => {
    res.send("hello from community")
})

router.get("/hub", (req,res) => {
    res.send("hello from Hub")
})

module.exports = router