import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load model and encoders ───────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("le_goal.pkl", "rb") as f:
        le_goal = pickle.load(f)
    with open("le_exp.pkl", "rb") as f:
        le_exp = pickle.load(f)
    with open("le_plan.pkl", "rb") as f:
        le_plan = pickle.load(f)
    print("✅ Model and encoders loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/encoders: {e}")
    model = le_goal = le_exp = le_plan = None

# ─── Workout Plans ─────────────────────────────────────────────────────────────
WORKOUT_PLANS = {
    "FatLoss_Beginner": {
        "Day 1": "Brisk Walking (30 min), Bodyweight Squats (3×15), Push-Ups (3×10)",
        "Day 2": "Cycling (20 min), Dumbbell Rows (3×12), Plank (3×30 sec)",
        "Day 3": "Rest / Light Stretching",
        "Day 4": "Jump Rope (15 min), Lunges (3×12), Dumbbell Shoulder Press (3×10)",
        "Day 5": "Jogging (25 min), Glute Bridges (3×15), Tricep Dips (3×10)",
        "Day 6": "Swimming or Aerobics (30 min), Core Circuit (3 rounds)",
        "Day 7": "Full Rest"
    },
    "FatLoss_Intermediate": {
        "Day 1": "Burpees (4×10), Bench Press (4×10), Goblet Squats (4×12)",
        "Day 2": "Running (30 min), Deadlift (3×8), Plank Variations (3×45 sec)",
        "Day 3": "HIIT Circuit (20 min), Pull-Ups (3×8), Mountain Climbers (3×20)",
        "Day 4": "Active Recovery – Yoga or Stretching",
        "Day 5": "Kettlebell Swings (4×15), Overhead Press (4×10), Romanian Deadlift (4×10)",
        "Day 6": "Cardio Intervals (25 min), Core Superset (4 rounds)",
        "Day 7": "Full Rest"
    },
    "FatLoss_Advanced": {
        "Day 1": "Barbell Complex (5×5), Box Jumps (4×10), Battle Ropes (3×30 sec)",
        "Day 2": "Sprint Intervals (30 min), Weighted Pull-Ups (4×6), Hanging Leg Raises (4×12)",
        "Day 3": "Olympic Lifting – Power Cleans (5×3), Front Squats (5×5)",
        "Day 4": "Active Recovery – Mobility Work",
        "Day 5": "Deadlift (5×5), Weighted Dips (4×8), TRX Core (4 rounds)",
        "Day 6": "CrossFit-style WOD (40 min), Loaded Carries",
        "Day 7": "Full Rest"
    },
    "MuscleGain_Beginner": {
        "Day 1": "Bench Press (3×10), Dumbbell Flyes (3×12), Push-Ups (3×15)",
        "Day 2": "Squat (3×10), Leg Press (3×12), Calf Raises (3×15)",
        "Day 3": "Rest",
        "Day 4": "Dumbbell Rows (3×10), Lat Pulldown (3×12), Bicep Curls (3×12)",
        "Day 5": "Overhead Press (3×10), Lateral Raises (3×12), Tricep Pushdown (3×12)",
        "Day 6": "Full Body Light Circuit, Core Work",
        "Day 7": "Full Rest"
    },
    "MuscleGain_Intermediate": {
        "Day 1": "Bench Press (4×8), Incline DB Press (4×10), Cable Flyes (3×12)",
        "Day 2": "Squat (4×8), Romanian Deadlift (3×10), Leg Curls (3×12)",
        "Day 3": "Pull-Ups (4×8), Barbell Row (4×8), Face Pulls (3×15)",
        "Day 4": "Rest / Mobility",
        "Day 5": "Overhead Press (4×8), Arnold Press (3×10), Lateral Raises (4×12)",
        "Day 6": "Deadlift (4×6), Farmer Walks, Core Circuit",
        "Day 7": "Full Rest"
    },
    "MuscleGain_Advanced": {
        "Day 1": "Heavy Bench Press (5×5), Weighted Dips (4×6), DB Flyes Drop Set",
        "Day 2": "Squat (5×5), Hack Squat (4×8), Bulgarian Split Squat (3×10)",
        "Day 3": "Weighted Pull-Ups (5×5), Pendlay Row (4×6), Straight-Arm Pulldown (3×15)",
        "Day 4": "Active Recovery – Foam Rolling, Stretching",
        "Day 5": "Overhead Press (5×5), Behind-Neck Press (3×8), Upright Rows (4×10)",
        "Day 6": "Deadlift (5×3), Rack Pulls, Strongman Accessories",
        "Day 7": "Full Rest"
    },
    "Maintenance_Beginner": {
        "Day 1": "Jogging (20 min), Bodyweight Squats (3×12), Push-Ups (3×10)",
        "Day 2": "Cycling (25 min), Dumbbell Rows (3×10), Plank (3×30 sec)",
        "Day 3": "Rest / Yoga",
        "Day 4": "Swimming (20 min), Lunges (3×10), Core Work",
        "Day 5": "Brisk Walk (30 min), Light Resistance Bands Circuit",
        "Day 6": "Recreational Sport or Dance (30 min)",
        "Day 7": "Full Rest"
    },
    "Maintenance_Intermediate": {
        "Day 1": "Running (30 min), Compound Lift – Squat or Bench (3×10)",
        "Day 2": "Full Body Resistance Training (moderate intensity)",
        "Day 3": "Yoga / Stretching / Mobility",
        "Day 4": "HIIT (20 min), Core Superset",
        "Day 5": "Cycling or Swimming (30 min), Upper Body Maintenance",
        "Day 6": "Recreational Activity or Sports",
        "Day 7": "Full Rest"
    },
    "Maintenance_Advanced": {
        "Day 1": "Strength Training – Push (5×5 moderate weight)",
        "Day 2": "Strength Training – Pull (5×5 moderate weight)",
        "Day 3": "Cardio + Mobility (40 min combined)",
        "Day 4": "Legs – Squat + Accessories (4×8)",
        "Day 5": "Functional Training – Olympic Lifts or Kettlebells",
        "Day 6": "Active Recovery – Sport or Long Walk",
        "Day 7": "Full Rest"
    }
}

# ─── Diet Plans ────────────────────────────────────────────────────────────────
DIET_PLANS = {
    "Cut": {
        "Goal": "Calorie Deficit – Fat Loss",
        "Calories": "~1,800–2,100 kcal/day (adjust by body weight)",
        "Macros": "Protein: 40% | Carbs: 35% | Fats: 25%",
        "Breakfast": "Egg white omelette with spinach + black coffee",
        "Mid-Morning": "Greek yogurt (low fat) + a handful of almonds",
        "Lunch": "Grilled chicken breast + steamed broccoli + brown rice (small portion)",
        "Evening Snack": "Whey protein shake + apple",
        "Dinner": "Baked salmon + roasted vegetables + quinoa",
        "Hydration": "3–4 litres of water daily",
        "Avoid": "Processed sugar, fried foods, alcohol, white bread"
    },
    "Bulk": {
        "Goal": "Calorie Surplus – Muscle Gain",
        "Calories": "~3,000–3,500 kcal/day (adjust by body weight)",
        "Macros": "Protein: 30% | Carbs: 50% | Fats: 20%",
        "Breakfast": "5 whole eggs + oats with banana + whole milk",
        "Mid-Morning": "Peanut butter toast + protein shake",
        "Lunch": "Rice + dal + paneer/chicken + vegetables",
        "Evening Snack": "Mass gainer shake + mixed nuts",
        "Dinner": "Pasta with ground beef/tofu + salad + bread",
        "Hydration": "3–4 litres of water daily",
        "Avoid": "Junk food, empty-calorie snacks"
    },
    "Maintain": {
        "Goal": "Balanced Nutrition – Weight Maintenance",
        "Calories": "~2,200–2,600 kcal/day (adjust by body weight)",
        "Macros": "Protein: 30% | Carbs: 45% | Fats: 25%",
        "Breakfast": "Whole grain cereal + milk + fruit",
        "Mid-Morning": "Mixed nuts + green tea",
        "Lunch": "Grilled protein (chicken/fish/tofu) + vegetables + rice or roti",
        "Evening Snack": "Protein bar or boiled eggs",
        "Dinner": "Lean meat or lentils + salad + whole grain bread",
        "Hydration": "2.5–3.5 litres of water daily",
        "Avoid": "Overeating, skipping meals, excess sugar"
    }
}

VALID_GOALS = {"Cut", "Bulk", "Maintain"}
VALID_EXP   = {"Beginner", "Intermediate", "Advanced"}

# ─── API Endpoint ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    # Validate required fields
    required = ["weight", "height", "goal", "experience"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: '{field}'"}), 400

    # Normalize strings (case-insensitive)
    goal       = str(data["goal"]).strip().capitalize()
    experience = str(data["experience"]).strip().capitalize()

    if goal not in VALID_GOALS:
        return jsonify({"error": f"Invalid goal '{goal}'. Choose: Cut, Bulk, Maintain"}), 400
    if experience not in VALID_EXP:
        return jsonify({"error": f"Invalid experience '{experience}'. Choose: Beginner, Intermediate, Advanced"}), 400

    try:
        weight = float(data["weight"])
        height = float(data["height"])
    except (ValueError, TypeError):
        return jsonify({"error": "Weight and height must be numbers."}), 400

    if not (20 <= weight <= 300):
        return jsonify({"error": "Weight must be between 20 and 300 kg."}), 400
    if not (100 <= height <= 250):
        return jsonify({"error": "Height must be between 100 and 250 cm."}), 400

    # Encode inputs
    try:
        goal_enc = le_goal.transform([goal])[0]
        exp_enc  = le_exp.transform([experience])[0]
    except Exception:
        return jsonify({"error": "Encoding error. Ensure encoders match training data."}), 500

    # Calculate BMI internally (model was trained with 5 features)
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)

    # Try with DataFrame first (matches feature names from training)
    try:
        import pandas as pd
        features = pd.DataFrame([[weight, height, bmi, goal_enc, exp_enc]],
                                 columns=["Weight", "Height", "BMI", "Goal", "Experience"])
        pred_label = model.predict(features)[0]
    except Exception:
        # Fallback: raw numpy array
        features = np.array([[weight, height, bmi, goal_enc, exp_enc]])
        pred_label = model.predict(features)[0]
    plan_name  = le_plan.inverse_transform([pred_label])[0]

    workout = WORKOUT_PLANS.get(plan_name, {"Error": "Plan not found in dictionary."})
    diet    = DIET_PLANS.get(goal, {})

    return jsonify({
        "plan":    plan_name,
        "workout": workout,
        "diet":    diet,
        "message": "Follow this plan consistently for 4 weeks for best results! 💪"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(debug=True, port=5000)