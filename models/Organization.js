import mongoose from "mongoose";

const OrganizationSchema = new mongoose.Schema({
  orgName: { type: String, required: true },
  email: { type: String, required: true, unique: true, lowercase: true },
  passwordHash: { type: String, required: true },
  passKey: { type: String, required: true, unique: true },
  greenScore: {type: Number, default: 0},
}, { timestamps: true });

export default mongoose.model("Organization", OrganizationSchema);
