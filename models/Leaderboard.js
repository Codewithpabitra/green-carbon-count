import mongoose from "mongoose";

// Leaderboard schema
const leaderboardSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  greenScore: {
    type: Number,
    default: 0
  },
  orgId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Organization'
  }
});

// Export the model
export default mongoose.model("Leaderboard", leaderboardSchema);
