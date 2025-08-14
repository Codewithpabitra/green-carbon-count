const express = require("express")
const path = require("path")
const navRouter = require("./routes/navigation.routes")
const authRouter = require("./routes/authentication.routes")
const dotenv = require("dotenv")
dotenv.config()
const { fileURLToPath } = require("url")
const app = express()

// const __filename = fileURLToPath(import.meta.url)
// const __dirname = path.dirname(__filename)

const PORT = process.env.PORT || 3000;

app.set('view engine', "ejs")
app.set("views",path.join(__dirname,"views"))

app.use(express.urlencoded({extended: true}))
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')));


app.use("/nav",navRouter) // now the route will be like this -> "/nav/users"
app.use("/users",authRouter) // now the route will be like this -> "/users/signup"

app.get("/",(req,res) => {
    res.render("home")
})

app.listen(3000, (req,res) => {
    console.log(`Server is connected to port ${PORT}`)
})







