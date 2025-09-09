"""
Medical Nutrition Dataset Creator using Real Public Datasets
Uses actual nutrition databases to create LLM training data

Required datasets to download:
1. USDA FoodData Central: https://fdc.nal.usda.gov/download-datasets.html
2. MyFoodData Database: https://www.myfooddata.com/
3. Recipe1M+ Dataset: http://pic2recipe.csail.mit.edu/
4. Food.com Recipe Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class RealNutritionDatasetProcessor:
    def __init__(self, data_dir: str = "./nutrition_data"):
        """
        Initialize with paths to real downloaded datasets
        
        Download these datasets first:
        1. USDA FoodData Central CSV files
        2. MyFoodData JSON files  
        3. Recipe datasets from Kaggle/academic sources
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        self.datasets = {}
        
    def load_usda_fooddata_central(self):
        """
        Load USDA FoodData Central database
        Download from: https://fdc.nal.usda.gov/download-datasets.html
        
        Files needed:
        - food.csv
        - nutrient.csv
        - food_nutrient.csv
        - food_category.csv
        """
        try:
            # Load main food database
            foods_df = pd.read_csv(self.data_dir / "FoodData_Central_csv_2024-04-18" / "food.csv")
            nutrients_df = pd.read_csv(self.data_dir / "FoodData_Central_csv_2024-04-18" / "nutrient.csv")
            food_nutrients_df = pd.read_csv(self.data_dir / "FoodData_Central_csv_2024-04-18" / "food_nutrient.csv")
            categories_df = pd.read_csv(self.data_dir / "FoodData_Central_csv_2024-04-18" / "food_category.csv")
            
            # Merge nutrition data
            nutrition_data = food_nutrients_df.merge(nutrients_df, on='nutrient_id', how='left')
            nutrition_data = nutrition_data.merge(foods_df, on='fdc_id', how='left')
            nutrition_data = nutrition_data.merge(categories_df, left_on='food_category_id', right_on='id', how='left')
            
            self.datasets['usda'] = nutrition_data
            print(f"Loaded USDA dataset: {len(nutrition_data)} nutrition records")
            return nutrition_data
            
        except FileNotFoundError:
            print("USDA FoodData Central files not found. Download from: https://fdc.nal.usda.gov/download-datasets.html")
            return None
        except pd.errors.EmptyDataError:
            print("One or more USDA CSV files are empty or corrupted.")
            return None
    
    def load_myfooddata_database(self):
        """
        Load MyFoodData nutrition database
        API: https://www.myfooddata.com/api/
        """
        try:
            # If you have the JSON export
            with open(self.data_dir / "myfooddata_export.json", 'r') as f:
                myfood_data = json.load(f)
            
            # Convert to DataFrame
            foods_list = []
            for food in myfood_data:
                foods_list.append({
                    'name': food.get('name'),
                    'calories': food.get('calories'),
                    'protein': food.get('protein'),
                    'carbs': food.get('carbs'),
                    'fat': food.get('fat'),
                    'fiber': food.get('fiber'),
                    'sugar': food.get('sugar'),
                    'sodium': food.get('sodium'),
                    'category': food.get('category')
                })
            
            myfood_df = pd.DataFrame(foods_list)
            self.datasets['myfooddata'] = myfood_df
            print(f"Loaded MyFoodData: {len(myfood_df)} foods")
            return myfood_df
            
        except FileNotFoundError:
            print("MyFoodData export not found. Use their API or download data")
            return None
        except json.JSONDecodeError:
            print("Invalid JSON format in MyFoodData export")
            return None
    
    def load_recipe_datasets(self):
        """
        Load recipe datasets from Kaggle/academic sources
        
        1. Food.com Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
        2. Recipe1M+: http://pic2recipe.csail.mit.edu/
        """
        recipes = {}
        
        try:
            # Load Food.com dataset (if downloaded)
            foodcom_recipes = pd.read_csv(self.data_dir / "RAW_recipes.csv")
            foodcom_interactions = pd.read_csv(self.data_dir / "RAW_interactions.csv")
            
            recipes['foodcom'] = {
                'recipes': foodcom_recipes,
                'interactions': foodcom_interactions
            }
            print(f"Loaded Food.com: {len(foodcom_recipes)} recipes")
            
        except FileNotFoundError:
            print("Food.com dataset not found. Download from Kaggle")
        
        try:
            # Load Recipe1M+ (if available)
            with open(self.data_dir / "recipe1m_layer1.json", 'r') as f:
                recipe1m = json.load(f)
            
            recipes['recipe1m'] = pd.DataFrame(recipe1m)
            print(f"Loaded Recipe1M+: {len(recipe1m)} recipes")
            
        except FileNotFoundError:
            print("Recipe1M+ dataset not found")
        
        self.datasets['recipes'] = recipes
        return recipes
    
    def load_medical_nutrition_guidelines(self):
        """
        Load medical nutrition guidelines from authoritative sources
        These should be scraped/compiled from:
        - American Diabetes Association
        - American Heart Association  
        - Academy of Nutrition and Dietetics
        - NIH dietary guidelines
        """
        guidelines = {
            'diabetes': {
                'carb_counting': '45-60g per meal',
                'protein': '15-20% calories',
                'fiber': '25-35g daily',
                'sodium': '<2300mg daily',
                'glycemic_index': 'low preferred'
            },
            'hypertension': {
                'sodium': '<1500mg daily',
                'potassium': '3500-4700mg daily',
                'dash_diet': True
            },
            'heart_disease': {
                'saturated_fat': '<7% calories',
                'omega3': '1-2g daily',
                'fiber': '25-35g daily'
            },
            'kidney_disease': {
                'protein': '0.8g/kg body weight',
                'phosphorus': '800-1000mg daily',
                'potassium': '2000-3000mg daily'
            }
        }
        
        self.datasets['medical_guidelines'] = guidelines
        return guidelines
    
    def create_training_examples(self):
        """
        Create LLM training examples using real data
        Format: {"input": "patient profile", "output": "diet plan"}
        """
        if 'usda' not in self.datasets:
            print("Load USDA dataset first")
            return None
        
        training_data = []
        
        # Sample patient profiles with real medical conditions
        sample_patients = [
            {
                "age": 45,
                "condition": "type_2_diabetes", 
                "allergies": ["peanuts"],
                "preferences": ["low_sugar", "high_protein"],
                "weight": 80,
                "activity": "sedentary"
            },
            {
                "age": 65,
                "condition": "hypertension",
                "allergies": [],
                "preferences": ["low_sodium", "heart_healthy"],
                "weight": 75,
                "activity": "lightly_active"
            },
            {
                "age": 15,
                "condition": "healthy",
                "allergies": ["peanuts"],
                "preferences": ["low_sugar", "high_protein"],
                "weight": 60,
                "activity": "very_active"
            }
        ]
        
        usda_data = self.datasets['usda']
        
        for patient in sample_patients:
            # Filter foods based on patient needs
            suitable_foods = self.filter_foods_for_patient(usda_data, patient)
            
            # Create meal plan using filtered foods
            meal_plan = self.generate_meal_plan(suitable_foods, patient)
            
            # Format as training example
            input_text = f"Age: {patient['age']}, Condition: {patient['condition']}, Allergies: {', '.join(patient['allergies'])}, Preferences: {', '.join(patient['preferences'])}"
            
            training_example = {
                "input": input_text,
                "output": meal_plan,
                "patient_profile": patient,
                "nutrition_analysis": self.analyze_nutrition(meal_plan)
            }
            
            training_data.append(training_example)
        
        return training_data
    
    def filter_foods_for_patient(self, food_data, patient_profile):
        """Filter foods based on patient's medical condition and allergies"""
        filtered_foods = food_data.copy()
        
        # Filter by allergies
        if patient_profile['allergies']:
            for allergen in patient_profile['allergies']:
                # Assuming food descriptions are in a column like 'description' in USDA data
                filtered_foods = filtered_foods[
                    ~filtered_foods['description'].str.lower().str.contains(allergen.lower(), na=False)
                ]
        
        # Filter based on medical condition (using guidelines from load_medical_nutrition_guidelines)
        condition = patient_profile['condition']
        guidelines = self.datasets.get('medical_guidelines', {}).get(condition, {})
        
        if condition == 'type_2_diabetes':
            # Filter for low sugar foods (assuming sugar is in nutrient data)
            filtered_foods = filtered_foods[
                (~filtered_foods['nutrient_name'].str.contains('Sugars', na=False)) |
                (filtered_foods['amount'] <= 5)  # Less than 5g sugar per 100g
            ]
        elif condition == 'hypertension':
            # Filter for low sodium foods
            filtered_foods = filtered_foods[
                (~filtered_foods['nutrient_name'].str.contains('Sodium', na=False)) |
                (filtered_foods['amount'] <= 140)  # Less than 140mg sodium per 100g
            ]
        elif condition == 'heart_disease':
            # Filter for low saturated fat foods
            filtered_foods = filtered_foods[
                (~filtered_foods['nutrient_name'].str.contains('Fatty acids, total saturated', na=False)) |
                (filtered_foods['amount'] <= 2)  # Less than 2g saturated fat per 100g
            ]
        elif condition == 'kidney_disease':
            # Filter for low protein and low potassium foods
            filtered_foods = filtered_foods[
                (~filtered_foods['nutrient_name'].str.contains('Protein', na=False)) |
                (filtered_foods['amount'] <= 10)  # Less than 10g protein per 100g
            ]
            filtered_foods = filtered_foods[
                (~filtered_foods['nutrient_name'].str.contains('Potassium', na=False)) |
                (filtered_foods['amount'] <= 200)  # Less than 200mg potassium per 100g
            ]
        # No specific filtering for 'healthy' condition
        
        # Apply preferences (e.g., high protein, low sugar)
        for pref in patient_profile['preferences']:
            if pref == 'high_protein':
                filtered_foods = filtered_foods[
                    (filtered_foods['nutrient_name'] != 'Protein') |
                    (filtered_foods['amount'] >= 10)  # At least 10g protein per 100g
                ]
            elif pref == 'low_sugar':
                filtered_foods = filtered_foods[
                    (~filtered_foods['nutrient_name'].str.contains('Sugars', na=False)) |
                    (filtered_foods['amount'] <= 5)
                ]
            elif pref == 'low_sodium':
                filtered_foods = filtered_foods[
                    (~filtered_foods['nutrient_name'].str.contains('Sodium', na=False)) |
                    (filtered_foods['amount'] <= 140)
                ]
            elif pref == 'heart_healthy':
                filtered_foods = filtered_foods[
                    (~filtered_foods['nutrient_name'].str.contains('Fatty acids, total saturated', na=False)) |
                    (filtered_foods['amount'] <= 2)
                ]
        
        return filtered_foods
    
    def generate_meal_plan(self, suitable_foods, patient_profile):
        """Generate a simple meal plan based on filtered foods (stub implementation)"""
        # Sample a few foods to create a basic meal plan
        if suitable_foods.empty:
            return {"error": "No suitable foods found for the patient profile"}
        
        # Example: Select 3 random foods for a daily meal plan
        selected_foods = suitable_foods.sample(n=min(3, len(suitable_foods)), random_state=42)
        meal_plan = {
            "breakfast": selected_foods.iloc[0]['description'] if len(selected_foods) > 0 else "None",
            "lunch": selected_foods.iloc[1]['description'] if len(selected_foods) > 1 else "None",
            "dinner": selected_foods.iloc[2]['description'] if len(selected_foods) > 2 else "None"
        }
        return meal_plan
    
    def analyze_nutrition(self, meal_plan):
        """
        Analyze the nutritional content of a meal plan using USDA dataset or Nutritionix API.
        
        Args:
            meal_plan (dict): A dictionary with meal names (e.g., 'breakfast', 'lunch', 'dinner') 
                            and food descriptions.
        
        Returns:
            dict: Nutritional totals for calories, protein, carbs, fat, and sodium.
        """
        if 'usda' not in self.datasets:
            print("USDA dataset not loaded. Load USDA dataset first.")
            return {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "sodium": 0,
                "error": "USDA dataset not loaded"
            }
        
        usda_data = self.datasets['usda']
        nutrition_totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "sodium": 0.0,
            "missing_foods": []
        }
        
        # Nutrient IDs or names based on USDA FoodData Central (adjust as needed)
        nutrient_mapping = {
            "calories": "Energy",
            "protein": "Protein",
            "carbs": "Carbohydrate, by difference",
            "fat": "Total lipid (fat)",
            "sodium": "Sodium, Na"
        }
        
        # Iterate through each meal in the meal plan
        for meal, food_desc in meal_plan.items():
            if food_desc == "None" or not food_desc:
                continue
                
            # Try to find the food in USDA dataset
            food_match = usda_data[usda_data['description'].str.lower() == food_desc.lower()]
            
            if not food_match.empty:
                # Sum nutritional values for the matched food
                for nutrient_key, nutrient_name in nutrient_mapping.items():
                    nutrient_data = food_match[food_match['nutrient_name'] == nutrient_name]
                    if not nutrient_data.empty:
                        amount = nutrient_data['amount'].iloc[0]  # Assume amount is per 100g
                        nutrition_totals[nutrient_key] += float(amount)  # Add to total (assuming 100g serving)
            else:
                # Food not found in USDA dataset
                nutrition_totals['missing_foods'].append(food_desc)
                print(f"Food '{food_desc}' not found in USDA dataset.")
        
        # If any foods were not found, try Nutritionix API (optional)
        if nutrition_totals['missing_foods'] and self._try_nutritionix_api:
            print("Attempting to fetch missing foods from Nutritionix API...")
            for food_desc in nutrition_totals['missing_foods']:
                nutrient_data = self._query_nutritionix_api(food_desc)
                if nutrient_data:
                    for nutrient_key in nutrient_mapping.keys():
                        nutrition_totals[nutrient_key] += nutrient_data.get(nutrient_key, 0.0)
                    nutrition_totals['missing_foods'].remove(food_desc)  # Remove from missing list if found
        
        return nutrition_totals

    def _try_nutritionix_api(self):
        """
        Check if Nutritionix API credentials are available.
        Requires environment variables: NUTRITIONIX_APP_ID, NUTRITIONIX_API_KEY
        """
        import os
        return os.getenv('NUTRITIONIX_APP_ID') and os.getenv('NUTRITIONIX_API_KEY')

    def _query_nutritionix_api(self, food_desc):
        """
        Query Nutritionix API for nutritional data of a food item.
        Requires Nutritionix API credentials.
        
        Args:
            food_desc (str): Food description to query
        
        Returns:
            dict: Nutritional data or None if query fails
        """
        import requests
        import os
        
        app_id = os.getenv('NUTRITIONIX_APP_ID')
        api_key = os.getenv('NUTRITIONIX_API_KEY')
        
        if not app_id or not api_key:
            print("Nutritionix API credentials not set. Skipping API query.")
            return None
        
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": app_id,
            "x-app-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {"query": food_desc}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json().get('foods', [{}])[0]
            
            # Map Nutritionix response to our nutrient keys (adjust based on API response)
            return {
                "calories": data.get('nf_calories', 0.0),
                "protein": data.get('nf_protein', 0.0),
                "carbs": data.get('nf_total_carbohydrate', 0.0),
                "fat": data.get('nf_total_fat', 0.0),
                "sodium": data.get('nf_sodium', 0.0)
            }
        except requests.RequestException as e:
            print(f"Nutritionix API query failed for '{food_desc}': {e}")
            return None