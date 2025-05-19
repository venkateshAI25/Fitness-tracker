from flask import Blueprint, Flask, render_template, request, jsonify

diet_bp = Blueprint('diet', __name__, template_folder='../templates')

# Macronutrient calculations
def calculate_macros(weight, height, age, gender, activity_level, goal):
    # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor Equation
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Activity multiplier
    activity_multipliers = {
        'sedentary': 1.2,  # Little or no exercise
        'light': 1.375,    # Light exercise 1-3 days/week
        'moderate': 1.55,  # Moderate exercise 3-5 days/week
        'active': 1.725,   # Hard exercise 6-7 days/week
        'very_active': 1.9 # Very hard exercise & physical job or 2x training
    }
    
    # Calculate TDEE (Total Daily Energy Expenditure)
    tdee = bmr * activity_multipliers[activity_level]
    
    # Adjust calories based on goal
    if goal == 'loss':
        calories = tdee - 500  # 500 calorie deficit for weight loss
    elif goal == 'gain':
        calories = tdee + 500  # 500 calorie surplus for weight gain
    else:  # maintenance
        calories = tdee
    
    # Calculate macronutrients (protein, carbs, fats)
    if goal == 'gain':
        protein_g = weight * 2.2  # 2.2g per kg for muscle gain
        fat_g = weight * 1  # 1g per kg body weight
    elif goal == 'loss':
        protein_g = weight * 2.5  # 2.5g per kg to preserve muscle during deficit
        fat_g = weight * 0.8  # 0.8g per kg body weight
    else:
        protein_g = weight * 1.8  # 1.8g per kg for maintenance
        fat_g = weight * 0.8  # 0.8g per kg body weight
    
    # Remaining calories from carbs
    protein_cals = protein_g * 4  # 4 calories per gram of protein
    fat_cals = fat_g * 9  # 9 calories per gram of fat
    carb_cals = calories - protein_cals - fat_cals
    carb_g = carb_cals / 4  # 4 calories per gram of carbohydrates
    
    return {
        'calories': round(calories),
        'protein': round(protein_g),
        'carbs': round(carb_g),
        'fat': round(fat_g)
    }

# Food suggestions based on goal
def get_food_suggestions(goal, dietary_preference=None):
    # Base food suggestions
    protein_sources = {
        'all': ['Chicken breast', 'Turkey', 'Lean beef', 'Fish (salmon, tuna, cod)', 'Eggs', 'Greek yogurt', 'Cottage cheese', 'Whey protein'],
        'vegetarian': ['Eggs', 'Greek yogurt', 'Cottage cheese', 'Whey protein', 'Tofu', 'Tempeh', 'Seitan', 'Legumes', 'Quinoa'],
        'vegan': ['Tofu', 'Tempeh', 'Seitan', 'Legumes (beans, lentils, chickpeas)', 'Nutritional yeast', 'Plant protein powder', 'Quinoa', 'Nuts and seeds']
    }
    
    carb_sources = {
        'all': ['Brown rice', 'Quinoa', 'Sweet potatoes', 'Oats', 'Whole grain bread', 'Whole grain pasta', 'Fruits', 'Vegetables'],
        'vegetarian': ['Brown rice', 'Quinoa', 'Sweet potatoes', 'Oats', 'Whole grain bread', 'Whole grain pasta', 'Fruits', 'Vegetables'],
        'vegan': ['Brown rice', 'Quinoa', 'Sweet potatoes', 'Oats', 'Whole grain bread', 'Whole grain pasta', 'Fruits', 'Vegetables']
    }
    
    fat_sources = {
        'all': ['Avocados', 'Olive oil', 'Nuts (almonds, walnuts)', 'Seeds (chia, flax)', 'Nut butters', 'Fatty fish', 'Eggs'],
        'vegetarian': ['Avocados', 'Olive oil', 'Nuts (almonds, walnuts)', 'Seeds (chia, flax)', 'Nut butters', 'Eggs', 'Cheese'],
        'vegan': ['Avocados', 'Olive oil', 'Nuts (almonds, walnuts)', 'Seeds (chia, flax)', 'Nut butters', 'Coconut']
    }
    
    # Select appropriate food sources based on dietary preference
    if dietary_preference not in protein_sources:
        dietary_preference = 'all'
    
    proteins = protein_sources[dietary_preference]
    carbs = carb_sources[dietary_preference]
    fats = fat_sources[dietary_preference]
    
    # Adjust suggestions based on goal
    if goal == 'loss':
        return {
            'proteins': proteins,
            'carbs': [item for item in carbs if item not in ['Whole grain pasta', 'Whole grain bread']],  # Limit higher GI carbs
            'fats': fats,
            'tips': [
                "Focus on high protein foods to preserve muscle mass",
                "Choose fibrous carbs that keep you full longer",
                "Aim for 3-4 meals per day with protein in each meal",
                "Stay hydrated - drink at least 2-3 liters of water daily",
                "Consider intermittent fasting (16:8) if it suits your lifestyle",
                "Limit processed foods and added sugars",
                "Include vegetables in most meals for nutrients and fiber"
            ]
        }
    elif goal == 'gain':
        return {
            'proteins': proteins,
            'carbs': carbs,
            'fats': fats,
            'tips': [
                "Eat 4-6 smaller meals throughout the day",
                "Include a protein and carb source in each meal",
                "Consume a post-workout meal with protein and carbs within 30-60 minutes",
                "Consider liquid calories (smoothies) if struggling to meet calorie goals",
                "Choose nutrient-dense foods to support recovery and muscle growth",
                "Stay hydrated - drink at least 2-3 liters of water daily",
                "Get adequate sleep (7-9 hours) to support recovery"
            ]
        }
    else:  # maintenance
        return {
            'proteins': proteins,
            'carbs': carbs,
            'fats': fats,
            'tips': [
                "Maintain balanced meals with protein, healthy carbs, and fats",
                "Focus on whole, unprocessed foods",
                "Stay consistent with meal timing",
                "Adjust portions based on hunger and activity levels",
                "Stay hydrated - drink at least 2-3 liters of water daily",
                "Include a variety of foods to ensure micronutrient intake"
            ]
        }

# Create sample meal plans
def create_meal_plan(macros, food_suggestions, goal):
    meal_count = 4 if goal == 'gain' else 3  # More meals for bulking
    
    meal_plans = []
    
    if goal == 'loss':
        meal_plans = [
            {
                'meal': 'Breakfast',
                'options': [
                    'Greek yogurt with berries and a tablespoon of honey',
                    'Vegetable omelette (2-3 eggs) with spinach and mushrooms',
                    'Overnight oats with protein powder and fruit',
                    'Protein smoothie with greens, berries, and a tablespoon of nut butter'
                ]
            },
            {
                'meal': 'Lunch',
                'options': [
                    'Grilled chicken salad with olive oil dressing',
                    'Tuna wrap with whole grain tortilla and vegetables',
                    'Turkey and vegetable soup with a side of whole grain bread',
                    'Quinoa bowl with roasted vegetables and lean protein'
                ]
            },
            {
                'meal': 'Dinner',
                'options': [
                    'Baked fish with steamed vegetables and quinoa',
                    'Stir-fried tofu or chicken with vegetables and brown rice',
                    'Lean beef or tempeh with sweet potato and green beans',
                    'Lentil curry with cauliflower rice'
                ]
            },
            {
                'meal': 'Snacks (choose 1-2)',
                'options': [
                    'Apple with a tablespoon of nut butter',
                    'Greek yogurt with a small handful of nuts',
                    'Protein shake with water',
                    'Carrot sticks with hummus',
                    'Hard-boiled egg'
                ]
            }
        ]
    elif goal == 'gain':
        meal_plans = [
            {
                'meal': 'Breakfast',
                'options': [
                    'Oatmeal made with milk, banana, protein powder, and nut butter',
                    '3-4 egg omelette with vegetables, avocado and whole grain toast',
                    'Greek yogurt parfait with granola, berries, and honey',
                    'Protein pancakes with fruit and maple syrup'
                ]
            },
            {
                'meal': 'Lunch',
                'options': [
                    'Chicken or tofu wrap with avocado and vegetables',
                    'Tuna or chickpea salad sandwich on whole grain bread',
                    'Rice bowl with lean protein, vegetables, and sauce',
                    'Pasta with meat or bean sauce and a side salad'
                ]
            },
            {
                'meal': 'Dinner',
                'options': [
                    'Salmon or tempeh with roasted vegetables and quinoa',
                    'Lean beef or lentil stir fry with rice',
                    'Chicken or tofu curry with rice and vegetables',
                    'Turkey or bean chili with corn bread'
                ]
            },
            {
                'meal': 'Snacks (choose 2-3)',
                'options': [
                    'Protein shake with milk, banana, and nut butter',
                    'Trail mix with nuts, dried fruit, and dark chocolate',
                    'Cottage cheese with pineapple',
                    'Rice cakes with nut butter and banana',
                    'Smoothie with yogurt, fruit, and oats',
                    'Tuna or hummus on whole grain crackers'
                ]
            }
        ]
    else:  # maintenance
        meal_plans = [
            {
                'meal': 'Breakfast',
                'options': [
                    'Oatmeal with fruit and nuts',
                    'Whole grain toast with eggs and avocado',
                    'Protein smoothie with spinach, fruit, and yogurt',
                    'Breakfast burrito with eggs or tofu, vegetables, and salsa'
                ]
            },
            {
                'meal': 'Lunch',
                'options': [
                    'Grain bowl with quinoa, vegetables, and protein of choice',
                    'Wrap with hummus, vegetables, and protein',
                    'Soup and salad combo with protein',
                    'Leftovers from dinner with added vegetables'
                ]
            },
            {
                'meal': 'Dinner',
                'options': [
                    'Baked protein (fish, chicken, tofu) with roasted vegetables',
                    'Stir fry with lean protein and lots of vegetables',
                    'Homemade burger (beef, turkey, or bean) with salad',
                    'Sheet pan dinner with protein and colorful vegetables'
                ]
            },
            {
                'meal': 'Snacks (choose 1-2)',
                'options': [
                    'Fresh fruit with a small handful of nuts',
                    'Vegetable sticks with hummus',
                    'Greek yogurt with berries',
                    'Small protein shake',
                    'Hard-boiled egg'
                ]
            }
        ]
    
    return meal_plans

@diet_bp.route('/')
def index():
    return render_template('diet.html')

@diet_bp.route('/get_diet_suggestions', methods=['POST'])
def get_diet_suggestions():
    # Get form data
    weight = float(request.form['weight'])
    height = float(request.form.get('height', 170))  # Default height if not provided
    age = int(request.form.get('age', 30))  # Default age if not provided
    gender = request.form.get('gender', 'male')  # Default gender if not provided
    activity_level = request.form.get('activity_level', 'moderate')  # Default activity if not provided
    goal = request.form['goal']
    dietary_preference = request.form.get('dietary_preference', 'all')  # Default diet preference if not provided
    
    # Calculate macros
    macros = calculate_macros(weight, height, age, gender, activity_level, goal)
    
    # Get food suggestions
    food_suggestions = get_food_suggestions(goal, dietary_preference)
    
    # Create meal plan
    meal_plan = create_meal_plan(macros, food_suggestions, goal)
    
    # Convert weight goal to more readable format
    goal_text = "Weight Loss" if goal == "loss" else "Weight Gain" if goal == "gain" else "Weight Maintenance"
    
    return render_template(
        'diet.html',
        weight=weight,
        height=height,
        age=age,
        gender=gender,
        activity_level=activity_level,
        goal=goal_text,
        macros=macros,
        food_suggestions=food_suggestions,
        meal_plan=meal_plan,
        dietary_preference=dietary_preference
    )