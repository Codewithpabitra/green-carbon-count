import express from "express"
import mongoose from "mongoose"
import path from 'path'
import navRouter from "./routes/navigation.routes.js"
import authRouter from "./routes/authentication.routes.js"
import orgRoutes from "./routes/org.js"
import studentRoutes from './routes/student.js'
import protectedRoutes from "./routes/protected.js"
import dotenv from "dotenv"
import cors from "cors";
import cookieParser from "cookie-parser";



dotenv.config()
import { fileURLToPath } from "url"
const app = express()

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PORT = process.env.PORT || 3000;

app.set('view engine', "ejs")
app.set("views",path.join(__dirname,"views"))

app.use(express.urlencoded({extended: true}))
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')));
app.use(cookieParser());

app.use(cors({
  origin: `http://localhost:${PORT}`, // frontend URL
  credentials: true,
  allowedHeaders: ["Content-Type", "Authorization"]
}));

mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log("✅ MongoDB connected"))
    .catch(err => console.error(err));


// app.use("/nav",navRouter) // now the route will be like this -> "/nav/users"
// app.use("/users",authRouter) // now the route will be like this -> "/users/signup"
app.use("/org", orgRoutes)
app.use("/student",studentRoutes)
app.use("/", protectedRoutes)



app.listen(PORT, (req,res) => {
    console.log(`Server is connected to port ${PORT}`)
})







