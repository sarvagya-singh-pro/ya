import requests
from datasets import load_dataset
import wikipedia


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
import re
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")
def evaluate_meal_nutrition(carb, fat, energy, protein, country="US"):
    guidelines = {
        "US": {"carb_max": 130, "carb_min": 45, "fat_max": 20, "fat_min": 5, "protein_min": 10, "energy_max": 800},
        "UK": {"carb_max": 260, "carb_min": 50, "fat_max": 70, "fat_min": 10, "protein_min": 15, "energy_max": 800},
        "ZMB": {"carb_max": 100, "carb_min": 40, "fat_max": 25, "fat_min": 8, "protein_min": 12, "energy_max": 700}
    }

    country_guidelines = guidelines.get(country, guidelines["US"])
    comments = []
    healthy = True

    if carb < country_guidelines["carb_min"]:
        comments.append(f"Low carbohydrate content for {country} guidelines, which might not provide enough energy.")
        healthy = False
    elif carb > country_guidelines["carb_max"]:
        comments.append(f"High carbohydrate content for {country} guidelines, which may lead to excess calorie intake.")
        healthy = False

    if fat < country_guidelines["fat_min"]:
        comments.append(f"Low fat content for {country} guidelines, which might lack essential fatty acids.")
        healthy = False
    elif fat > country_guidelines["fat_max"]:
        comments.append(f"High fat content for {country} guidelines, which might be unhealthy if mostly saturated or trans fats.")
        healthy = False

    if protein < country_guidelines["protein_min"]:
        comments.append(f"Low protein content for {country} guidelines, which may not support muscle and cell repair.")
        healthy = False

    if energy > country_guidelines["energy_max"]:
        comments.append(f"High calorie content for {country} guidelines; consider portion control.")
        healthy = False

    if healthy:
        comments.append(f"Meal aligns with balanced nutrition for {country} guidelines.")

    return " ".join(comments), healthy

def extract_foods(meal_description):
    prefixes = ["For breakfast, I ate", "For lunch, I ate", "For dinner, I had", "For a quick snack, I opted for"]
    
    # Fix the loop - need to iterate through prefixes
    for prefix in prefixes:
        meal_description = meal_description.replace(prefix, "").strip()
    
    foods = [food.strip() for food in meal_description.split(",") if food.strip()]
    cleaned_foods = []
    
    for food in foods:
        # Remove quantity patterns and descriptors
        for pattern in [r"\d+g", r"\(\d+g\)"]:
            food = re.sub(pattern, "", food).strip()
        
        for term in ["along with", "and", "boiled", "raw", "large", "ripe", "medium fat"]:
            food = food.replace(term, "").strip()
        
        # Clean up extra spaces and empty strings
        food = re.sub(r'\s+', ' ', food).strip()
        
        if food:
            cleaned_foods.append(food)
    
    return cleaned_foods

def get_nutrition_info_from_api(food_name):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": food_name}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        resp_json = response.json()
        
        if not resp_json or "foods" not in resp_json or not resp_json["foods"]:
            print(f"No valid food data returned for '{food_name}'")
            return {"serving_size": "Unknown", "ill_effects": [], "good_effects": []}
        
        food = resp_json["foods"][0]
        serving_size = f"{food.get('serving_qty', '')} {food.get('serving_unit', '')}".strip() or "Unknown"
        serving_weight = food.get("serving_weight_grams", 1) or 1
        
        ill_effects = []
        good_effects = []
        
        # Nutritional analysis
        saturated_fat = food.get("nf_saturated_fat", 0) or 0
        sugars = food.get("nf_sugars", 0) or 0
        fiber = food.get("nf_dietary_fiber", 0) or 0
        protein = food.get("nf_protein", 0) or 0
        potassium = food.get("nf_potassium", 0) or 0
        calcium = food.get("nf_calcium", 0) or 0
        iron = food.get("nf_iron", 0) or 0
        vitamin_c = food.get("nf_vitamin_c", 0) or 0
        
        # Negative effects
        if saturated_fat > (2.0 * serving_weight / 100):
            ill_effects.append("High in saturated fat relative to serving size.")
        
        if sugars > (5 * serving_weight / 100):
            ill_effects.append("High sugar content relative to serving size.")
        
        # Positive effects
        if fiber > (2 * serving_weight / 100):
            good_effects.append("Good source of dietary fiber for digestive health.")
        
        if protein > (5 * serving_weight / 100):
            good_effects.append("Contains protein for muscle building and repair.")
        
        if potassium > (200 * serving_weight / 100):
            good_effects.append("Rich in potassium for heart and muscle function.")
        
        if calcium > (100 * serving_weight / 100):
            good_effects.append("Contains calcium for bone and teeth health.")
        
        if iron > (1 * serving_weight / 100):
            good_effects.append("Source of iron for blood and oxygen transport.")
        
        if vitamin_c > (10 * serving_weight / 100):
            good_effects.append("Contains vitamin C for immune system support.")
        
        return {"serving_size": serving_size, "ill_effects": ill_effects, "good_effects": good_effects}
    
    except Exception as e:
        print(f"Error fetching nutrition info for '{food_name}': {e}")
        return {"serving_size": "Unknown", "ill_effects": [], "good_effects": []}

def get_nutrient_info_from_wikipedia(nutrient_name):
    """Get information about a nutrient from Wikipedia"""
    try:
        # Search for the nutrient page
        search_results = wikipedia.search(nutrient_name, results=1)
        if not search_results:
            return f"{nutrient_name}: No information available."
        
        # Get the page summary
        summary = wikipedia.summary(search_results[0], sentences=2)
        return f"{nutrient_name}: {summary}"
    
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            # Try the first option if there's disambiguation
            summary = wikipedia.summary(e.options[0], sentences=2)
            return f"{nutrient_name}: {summary}"
        except:
            return f"{nutrient_name}: Multiple topics found, information unclear."
    
    except Exception as e:
        return f"{nutrient_name}: Unable to retrieve information ({str(e)})."
def get_macronutrient_info():
    """Get information about macronutrients"""
    nutrients = {
        "Carbohydrates": "carbohydrate nutrition",
        "Protein": "protein nutrition", 
        "Fat": "dietary fat",
        "Energy": "food energy"
    }

    nutrient_info = {}
    for nutrient, search_term in nutrients.items():
        nutrient_info[nutrient] = get_nutrient_info_from_wikipedia(search_term)

    return nutrient_info

def generate_chain_of_thought(meal_description, carb, fat, energy, protein, country):
    # Get macronutrient information from Wikipedia
    nutrient_info_dict = get_macronutrient_info()
    
    nutrient_info = f"Carbohydrates: {carb}g, Fat: {fat}g, Energy: {energy}kcal, Protein: {protein}g."
    local_eval, healthy = evaluate_meal_nutrition(carb, fat, energy, protein, country)
    
    foods = extract_foods(meal_description)
    api_results = []
    
    for food in foods:
        if food:  # Only process non-empty food names
            api_info = get_nutrition_info_from_api(food)
            ill_effects_text = ' '.join(api_info['ill_effects']) if api_info['ill_effects'] else 'None'
            good_effects_text = ' '.join(api_info['good_effects']) if api_info['good_effects'] else 'None'
            api_results.append(f"{food}: Serving size: {api_info['serving_size']}, Benefits: {good_effects_text}, Concerns: {ill_effects_text}")
    
    api_info_text = "; ".join(api_results) if api_results else "No API data available."
    
    # Create nutrient information section
    nutrient_explanations = "\n".join([f"• {info}" for info in nutrient_info_dict.values()])
    
    cot = (
        f"Nutritional summary: {nutrient_info}\n\n"
        f"What these nutrients do:\n{nutrient_explanations}\n\n"
        f"Food analysis: {api_info_text}\n\n"
        f"Meal evaluation: {local_eval}"
    )
    return cot

def main():
    try:
        dataset = load_dataset("dongx1997/NutriBench", "v2", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    for i, example in enumerate(dataset):
        try:
            meal_desc = example['meal_description']
            carb = example['carb']
            fat = example['fat']
            energy = example['energy']
            protein = example['protein']
            country = example.get('country', 'US')
            
            print(f"Meal Description: {meal_desc}\n")
            chain_of_thought = generate_chain_of_thought(meal_desc, carb, fat, energy, protein, country)
            print("Chain-of-Thought:")
            print(chain_of_thought)
            print("=" * 100)
            
        except Exception as e:
            print(f"Error processing meal {i}: {e}")
        
        if i >= 4:
            break
import json

def save_output_to_jsonl(meal_description, carb, fat, energy, protein, country, filename="meal_analysis.jsonl"):
    nutrient_info_dict = get_macronutrient_info()
    nutrient_info = f"Carbohydrates: {carb}g, Fat: {fat}g, Energy: {energy}kcal, Protein: {protein}g."
    evaluation, healthy = evaluate_meal_nutrition(carb, fat, energy, protein, country)
    foods = extract_foods(meal_description)

    food_infos = []
    for food in foods:
        info = get_nutrition_info_from_api(food)
        food_infos.append({
            "food": food,
            "serving_size": info.get("serving_size", "Unknown"),
            "ill_effects": info.get("ill_effects", []),
            "good_effects": info.get("good_effects", [])
        })

    data = {
        "meal_description": meal_description,
        "country": country,
        "nutrient_summary": nutrient_info,
        "evaluation": evaluation,
        "is_healthy": healthy,
        "foods": food_infos,
        "macronutrient_wikipedia_info": nutrient_info_dict
    }

    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")
if __name__ == "__main__":
    main()
    save_output_to_jsonl(meal_description, carb, fat, energy, protein, country)