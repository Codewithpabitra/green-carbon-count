import mongoose from "mongoose";

const StudentSchema = new mongoose.Schema({
  orgId: { type: mongoose.Schema.Types.ObjectId, ref: "Organization", required: true },
  name: { type: String, required: true },
  passwordHash: { type: String, required: true },
  greenScore: {type: Number, default: 0},
}, { timestamps: true });

export default mongoose.model("Student", StudentSchema); 