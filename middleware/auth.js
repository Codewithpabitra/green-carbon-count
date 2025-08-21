import jwt from "jsonwebtoken";
import dotenv from "dotenv"


dotenv.config()

// Verify token exists + valid
// export const authenticate = (req, res, next) => {
  
//   console.log("Authorization Header:", req.headers["authorization"]);

//   const token = req.headers["authorization"]?.split(" ")[1]; // Bearer <token>
//   if (!token) return res.status(401).send("Access denied. No token provided.");

//   try {
//     const decoded = jwt.verify(token, process.env.JWT_SECRET);
//     req.user = decoded; // contains {id, role, orgId?, orgName?}
//     next();
//   } catch (err) {
//     res.status(401).send("Invalid token.");
//   }
// };


export const authenticate = (req, res, next) => {
  const token = req.cookies.token; // ✅ JWT comes from cookie
  if (!token) return res.status(401).send("Access denied");

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(401).send("Invalid token.");
  }
};

// Allow only organization admins
export const isOrg = (req, res, next) => {
  if (req.user?.role !== "org") return res.status(403).send("Only organization admins can access this.");
  next();
};

// Allow only students
export const isStudent = (req, res, next) => {
  if (req.user?.role !== "student") return res.status(403).send("Only students can access this.");
  next();
};
