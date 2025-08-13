const express = require("express")
const navRouter = require("./routes/navigation.routes")
const app = express()

app.set('view engine', "ejs")
app.use(express.urlencoded({extended: true}))
app.use(express.json())

app.use("/nav",navRouter)

app.get("/",(req,res) => {
    res.render("home")
})

app.listen(3000, (req,res) => {
    console.log("Server is connected")
})







