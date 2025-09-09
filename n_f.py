# ==============================================
# 🏥 ENHANCED UNIFIED HEALTHCARE AI SYSTEM
# Clinical Decision Support + Nutritional Analysis + Confidence Scoring
# ==============================================

# ==============================================
# ⚙️ SETUP ENVIRONMENT (Enhanced)
# ==============================================

import os
import json
import torch
import pandas as pd
import numpy as np
import requests
import wikipedia
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import math

# ML/AI Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from huggingface_hub import login

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# ==============================================
# 🎯 CONFIDENCE SCORING SYSTEM
# ==============================================
class ConfidenceScorer:
    """
    Advanced confidence scoring system for healthcare AI recommendations
    """
    
    def __init__(self):
        self.confidence_weights = {
            'model_confidence': 0.35,      # Model's internal confidence
            'response_coherence': 0.20,    # How coherent the response is
            'medical_validity': 0.25,      # Medical validity checks
            'safety_score': 0.20           # Safety assessment score
        }
        
        # Medical entity patterns for validation
        self.medical_patterns = {
            'medications': r'\b(?:mg|mcg|ml|tablet|capsule|injection|daily|twice|thrice)\b',
            'dosages': r'\b\d+\s*(?:mg|mcg|ml|g|kg|units?)\b',
            'conditions': r'\b(?:diabetes|hypertension|cancer|infection|pain|fever)\b',
            'lab_values': r'\b\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|%|bpm)\b'
        }
        
        # Initialize calibration parameters
        self.calibration_params = {
            'temperature': 1.0,
            'bias': 0.0
        }
    
    def calculate_model_confidence(self, 
                                   model_outputs: Dict,
                                   generation_params: Dict) -> Tuple[float, Dict]:
        """
        Calculate model's internal confidence based on generation probabilities
        """
        confidence_metrics = {}
        
        # 1. Token-level confidence (from logits)
        if 'logits' in model_outputs and model_outputs['logits'] is not None:
            logits = model_outputs['logits']
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate entropy (lower entropy = higher confidence)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_entropy = torch.mean(entropy).item()
            
            # Normalize entropy to confidence score (0-1)
            # Ensure probs.shape[-1] is not zero to avoid division by zero
            max_entropy = math.log(probs.shape[-1]) if probs.shape[-1] > 0 else 1.0 # Maximum possible entropy
            entropy_confidence = 1.0 - (avg_entropy / max_entropy)
            confidence_metrics['entropy_confidence'] = entropy_confidence
        else:
            confidence_metrics['entropy_confidence'] = 0.5  # Default moderate confidence
        
        # 2. Top-k probability confidence
        if 'sequences_scores' in model_outputs and model_outputs['sequences_scores'] is not None:
            # Use sequence scores if available
            scores = model_outputs['sequences_scores']
            max_score = torch.max(scores).item()
            prob_confidence = torch.sigmoid(torch.tensor(max_score)).item()
            confidence_metrics['probability_confidence'] = prob_confidence
        else:
            confidence_metrics['probability_confidence'] = 0.5
        
        # 3. Generation parameters influence
        temp_penalty = 1.0 - min(generation_params.get('temperature', 0.7), 1.0)
        top_p_boost = generation_params.get('top_p', 0.9)
        param_confidence = (temp_penalty + top_p_boost) / 2
        confidence_metrics['parameter_confidence'] = param_confidence
        
        # 4. Response length confidence (very short or very long responses less confident)
        response_length = len(model_outputs.get('generated_text', '').split())
        if 10 <= response_length <= 100:
            length_confidence = 1.0
        elif response_length < 5:
            length_confidence = 0.3
        elif response_length > 200:
            length_confidence = 0.6
        else:
            length_confidence = 0.8
        confidence_metrics['length_confidence'] = length_confidence
        
        # Combine all model confidence metrics
        model_confidence = np.mean([
            confidence_metrics['entropy_confidence'],
            confidence_metrics['probability_confidence'],
            confidence_metrics['parameter_confidence'],
            confidence_metrics['length_confidence']
        ])
        
        return model_confidence, confidence_metrics
    
    def calculate_response_coherence(self, response: str, query: str) -> Tuple[float, Dict]:
        """
        Calculate response coherence and relevance
        """
        coherence_metrics = {}
        
        # 1. Response completeness
        response_words = response.split()
        if len(response_words) < 5:
            completeness = 0.2
        elif len(response_words) < 20:
            completeness = 0.6
        else:
            completeness = 1.0
        coherence_metrics['completeness'] = completeness
        
        # 2. Repetition penalty
        unique_words = set(response_words)
        if len(response_words) > 0:
            repetition_ratio = len(unique_words) / len(response_words)
        else:
            repetition_ratio = 0
        coherence_metrics['repetition_score'] = repetition_ratio
        
        # 3. Medical terminology presence
        medical_terms = 0
        for pattern in self.medical_patterns.values():
            medical_terms += len(re.findall(pattern, response.lower()))
        
        medical_density = min(medical_terms / max(len(response_words), 1), 1.0)
        coherence_metrics['medical_terminology'] = medical_density
        
        # 4. Query relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words_set = set(response.lower().split())
        
        if len(query_words) > 0:
            relevance = len(query_words.intersection(response_words_set)) / len(query_words)
        else:
            relevance = 0
        coherence_metrics['relevance'] = relevance
        
        # 5. Structural coherence (presence of proper sentence structure)
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        structural_score = min(len(valid_sentences) / max(len(sentences), 1), 1.0)
        coherence_metrics['structural_coherence'] = structural_score
        
        # Combine coherence metrics
        coherence_score = np.mean([
            completeness * 0.25,
            repetition_ratio * 0.20,
            medical_density * 0.20,
            relevance * 0.20,
            structural_score * 0.15
        ])
        
        return coherence_score, coherence_metrics
    
    def calculate_medical_validity(self, 
                                   response: str, 
                                   domain: str, 
                                   patient_info: Dict = None) -> Tuple[float, Dict]:
        """
        Calculate medical validity score
        """
        validity_metrics = {}
        
        # 1. Domain-appropriate terminology
        response_lower = response.lower()
        
        if domain == "clinical":
            clinical_terms = [
                'treatment', 'medication', 'diagnosis', 'symptoms', 'patient',
                'therapy', 'prescription', 'dosage', 'side effect', 'contraindication'
            ]
            term_presence = sum(1 for term in clinical_terms if term in response_lower)
            terminology_score = min(term_presence / len(clinical_terms), 1.0)
        elif domain == "nutrition":
            nutrition_terms = [
                'calories', 'protein', 'carbohydrate', 'fat', 'vitamin',
                'nutrient', 'diet', 'healthy', 'balanced', 'portion'
            ]
            term_presence = sum(1 for term in nutrition_terms if term in response_lower)
            terminology_score = min(term_presence / len(nutrition_terms), 1.0)
        else:
            terminology_score = 0.5
        
        validity_metrics['terminology_score'] = terminology_score
        
        # 2. Dosage and measurement validity
        dosages = re.findall(self.medical_patterns['dosages'], response)
        valid_dosages = 0
        for dosage in dosages:
            # Simple validation - check if dosage is within reasonable ranges
            numbers = re.findall(r'\d+(?:\.\d+)?', dosage)
            if numbers:
                value = float(numbers[0])
                if 0.1 <= value <= 2000:  # Reasonable medical dosage range
                    valid_dosages += 1
        
        dosage_validity = valid_dosages / max(len(dosages), 1) if dosages else 1.0
        validity_metrics['dosage_validity'] = dosage_validity
        
        # 3. Logical medical flow
        medical_flow_patterns = [
            r'diagnos\w+.*treat\w+',
            r'symptom\w+.*medication',
            r'condition.*management',
            r'first.line.*second.line'
        ]
        
        flow_score = 0
        for pattern in medical_flow_patterns:
            if re.search(pattern, response_lower):
                flow_score += 1
        
        logical_flow = min(flow_score / len(medical_flow_patterns), 1.0)
        validity_metrics['logical_flow'] = logical_flow
        
        # 4. Patient-specific appropriateness
        if patient_info:
            appropriateness_score = self._assess_patient_appropriateness(
                response, patient_info
            )
        else:
            appropriateness_score = 0.8  # Default when no patient info
        
        validity_metrics['patient_appropriateness'] = appropriateness_score
        
        # Combine validity metrics
        medical_validity = np.mean([
            terminology_score * 0.3,
            dosage_validity * 0.25,
            logical_flow * 0.25,
            appropriateness_score * 0.2
        ])
        
        return medical_validity, validity_metrics
    
    def _assess_patient_appropriateness(self, response: str, patient_info: Dict) -> float:
        """Assess if response is appropriate for patient demographics"""
        score = 1.0
        response_lower = response.lower()
        
        # Age-based appropriateness
        age = patient_info.get('age', 0)
        if age > 65:
            # Check for elderly-inappropriate medications
            risky_for_elderly = ['benzodiazepine', 'anticholinergic', 'high dose']
            for risk in risky_for_elderly:
                if risk in response_lower:
                    score -= 0.2
        
        # Pregnancy considerations
        if patient_info.get('pregnant', False):
            pregnancy_risks = ['warfarin', 'ace inhibitor', 'high mercury']
            for risk in pregnancy_risks:
                if risk in response_lower:
                    score -= 0.3
        
        # Condition-specific appropriateness
        conditions = patient_info.get('conditions', [])
        if 'diabetes' in conditions and 'high sugar' in response_lower:
            score -= 0.25
        if 'hypertension' in conditions and 'high sodium' in response_lower:
            score -= 0.25
        
        return max(score, 0.0)
    
    def calculate_overall_confidence(self,
                                     model_outputs: Dict,
                                     response: str,
                                     query: str,
                                     domain: str,
                                     safety_assessment: Dict,
                                     patient_info: Dict = None,
                                     generation_params: Dict = None) -> Dict:
        """
        Calculate overall confidence score combining all metrics
        """
        if generation_params is None:
            generation_params = {'temperature': 0.7, 'top_p': 0.9}
        
        # Calculate individual confidence components
        model_conf, model_metrics = self.calculate_model_confidence(
            model_outputs, generation_params
        )
        
        coherence_conf, coherence_metrics = self.calculate_response_coherence(
            response, query
        )
        
        medical_conf, medical_metrics = self.calculate_medical_validity(
            response, domain, patient_info
        )
        
        # Safety score (inverse of risk level)
        safety_score = 1.0 - (safety_assessment.get('risk_level', 0) / 4.0)
        
        # Calculate weighted overall confidence
        overall_confidence = (
            model_conf * self.confidence_weights['model_confidence'] +
            coherence_conf * self.confidence_weights['response_coherence'] +
            medical_conf * self.confidence_weights['medical_validity'] +
            safety_score * self.confidence_weights['safety_score']
        )
        
        # Apply calibration
        calibrated_confidence = self._calibrate_confidence(overall_confidence)
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(calibrated_confidence)
        
        return {
            'overall_confidence': calibrated_confidence,
            'confidence_level': confidence_level,
            'component_scores': {
                'model_confidence': model_conf,
                'response_coherence': coherence_conf,
                'medical_validity': medical_conf,
                'safety_score': safety_score
            },
            'detailed_metrics': {
                'model_metrics': model_metrics,
                'coherence_metrics': coherence_metrics,
                'medical_metrics': medical_metrics
            },
            'uncertainty_quantification': self._quantify_uncertainty(
                calibrated_confidence, model_metrics
            )
        }
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score"""
        # Simple temperature scaling calibration
        calibrated = 1 / (1 + np.exp(-(raw_confidence - self.calibration_params['bias']) / 
                                     self.calibration_params['temperature']))
        return calibrated
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _quantify_uncertainty(self, confidence: float, model_metrics: Dict) -> Dict:
        """Quantify different types of uncertainty"""
        
        # Aleatoric uncertainty (data uncertainty)
        entropy_conf = model_metrics.get('entropy_confidence', 0.5)
        aleatoric_uncertainty = 1.0 - entropy_conf
        
        # Epistemic uncertainty (model uncertainty)
        prob_conf = model_metrics.get('probability_confidence', 0.5)
        epistemic_uncertainty = 1.0 - prob_conf
        
        # Total uncertainty
        total_uncertainty = 1.0 - confidence
        
        return {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence_interval': {
                'lower_bound': max(0.0, confidence - total_uncertainty),
                'upper_bound': min(1.0, confidence + total_uncertainty)
            }
        }

# ==============================================
# 🧠 ENHANCED UNIFIED HEALTHCARE AI MODEL
# ==============================================
class EnhancedUnifiedHealthcareAI:
    """
    Enhanced Unified Healthcare AI system with confidence scoring
    """

    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.clinical_pipeline = None
        self.nutrition_pipeline = None
        
        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer()

        # Configuration
        self.safety_rules = self._load_safety_rules()
        self.medical_guidelines = self._load_medical_guidelines()
        self.nutrition_standards = self._load_nutrition_standards()

        # Authentication
        self.auth_status = self._setup_authentication()

    def _setup_authentication(self):
        """Setup authentication for various services"""
        auth_status = {
            'huggingface': False,
            'wandb': False,
            'nutritionix': False
        }

        # Simple authentication setup
        try:
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                login(token=hf_token)
                auth_status['huggingface'] = True
                print("✅ Hugging Face authentication successful")
        except Exception as e:
            print(f"⚠️ Hugging Face authentication skipped: {e}")

        return auth_status

    def _load_safety_rules(self) -> Dict:
        """Load comprehensive safety rules for both domains"""
        return {
            'dangerous_drugs': [
                'thalidomide', 'rosiglitazone', 'rofecoxib', 'sibutramine',
                'phenylpropanolamine', 'ephedra', 'fen-phen'
            ],
            'contraindicated_combinations': [
                'warfarin + aspirin',
                'ace inhibitor + potassium',
                'monoamine oxidase inhibitor + tyramine',
                'calcium channel blocker + grapefruit'
            ],
            'dangerous_foods': [
                'raw shellfish for immunocompromised',
                'unpasteurized dairy for pregnant women',
                'high sodium for hypertensive patients'
            ],
            'allergy_triggers': [
                'peanuts', 'tree nuts', 'shellfish', 'eggs', 'milk', 'soy', 'wheat', 'fish'
            ]
        }

    def _load_medical_guidelines(self) -> Dict:
        """Load medical guidelines database"""
        return {
            'diabetes': {
                'first_line': 'metformin',
                'hba1c_target': '<7%',
                'lifestyle': 'diet modification, exercise, weight management'
            },
            'hypertension': {
                'first_line': 'ACE inhibitors or ARBs',
                'target_bp': '<140/90 mmHg',
                'lifestyle': 'DASH diet, sodium restriction, exercise'
            },
            'hyperlipidemia': {
                'first_line': 'statins',
                'ldl_target': '<100 mg/dL',
                'lifestyle': 'low saturated fat diet, exercise'
            }
        }

    def _load_nutrition_standards(self) -> Dict:
        """Load nutrition standards by country"""
        return {
            "US": {
                "carb_max": 130, "carb_min": 45,
                "fat_max": 20, "fat_min": 5,
                "protein_min": 10, "energy_max": 800
            },
            "UK": {
                "carb_max": 260, "carb_min": 50,
                "fat_max": 70, "fat_min": 10,
                "protein_min": 15, "energy_max": 800
            }
        }

    def load_model(self):
        """Load and configure the base model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
                output_attentions=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Successfully loaded {self.model_name}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _get_clinical_context(self, condition: str) -> str:
        """Get clinical context for given condition"""
        guidelines = {
            'diabetes': "ADA 2024: Metformin first-line, HbA1c target <7%, lifestyle modification essential",
            'hypertension': "AHA/ACC 2023: ACE inhibitors first-line, BP target <140/90, DASH diet recommended",
            'general': "Evidence-based medicine principles, patient safety first, individualized care"
        }
        return guidelines.get(condition, guidelines['general'])

    def _get_nutrition_context(self, country: str) -> str:
        """Get nutrition context for given country"""
        contexts = {
            'US': "USDA Dietary Guidelines 2020-2025: Balanced plate, limit added sugars and sodium",
            'UK': "NHS Eatwell Guide: 5-a-day fruits/vegetables, whole grains, lean proteins"
        }
        return contexts.get(country, contexts['US'])

# ==============================================
# 🔒 ENHANCED SAFETY SYSTEM
# ==============================================
class EnhancedSafetySystem:
    """Enhanced safety system with confidence-aware checks"""

    def __init__(self, safety_rules: Dict):
        self.safety_rules = safety_rules
        self.risk_levels = {
            'safe': 0,
            'caution': 1,
            'warning': 2,
            'danger': 3,
            'critical': 4
        }

    def comprehensive_safety_check(self,
                                   recommendation: str,
                                   domain: str,
                                   confidence_score: float,
                                   patient_info: Dict = None) -> Tuple[bool, str, int]:
        """Enhanced safety check considering confidence"""
        
        issues = []
        max_risk_level = 0

        # Adjust risk threshold based on confidence
        if confidence_score < 0.5:
            # Lower confidence = higher scrutiny
            risk_multiplier = 1.5
        elif confidence_score < 0.3:
            risk_multiplier = 2.0
        else:
            risk_multiplier = 1.0

        # Basic safety checks
        basic_safe, basic_msg, basic_risk = self._basic_safety_check(recommendation)
        if not basic_safe:
            issues.append(basic_msg)
            max_risk_level = max(max_risk_level, int(basic_risk * risk_multiplier))

        # Domain-specific safety checks
        domain_safe, domain_msg, domain_risk = self._domain_specific_safety_check(
            recommendation, domain, patient_info
        )
        if not domain_safe:
            issues.append(domain_msg)
            max_risk_level = max(max_risk_level, int(domain_risk * risk_multiplier))

        # Generate final assessment
        is_safe = max_risk_level <= 1
        safety_message = self._generate_safety_message(issues, max_risk_level, confidence_score)

        return is_safe, safety_message, max_risk_level

    def _basic_safety_check(self, recommendation: str) -> Tuple[bool, str, int]:
        """Basic safety checks"""
        rec_lower = recommendation.lower()

        for drug in self.safety_rules['dangerous_drugs']:
            if drug.lower() in rec_lower:
                return False, f"⛔ DANGEROUS DRUG: {drug} is contraindicated", 4
        
        for combo in self.safety_rules['contraindicated_combinations']:
            if all(term.strip() in rec_lower for term in combo.split('+')):
                return False, f"⛔ CONTRAINDICATED COMBINATION: {combo} detected", 3

        return True, "✅ Basic safety check passed", 0

    def _domain_specific_safety_check(self, recommendation: str, domain: str, patient_info: Dict) -> Tuple[bool, str, int]:
        """Perform domain-specific safety checks"""
        rec_lower = recommendation.lower()

        if domain == "nutrition" and patient_info:
            for food in self.safety_rules['dangerous_foods']:
                if food.lower() in rec_lower:
                    if 'immunocompromised' in food and patient_info.get('immunocompromised'):
                        return False, f"⛔ DANGEROUS FOOD: {food} for immunocompromised patient", 3
                    if 'pregnant women' in food and patient_info.get('pregnant'):
                        return False, f"⛔ DANGEROUS FOOD: {food} for pregnant patient", 3
                    if 'hypertensive patients' in food and 'hypertension' in patient_info.get('conditions', []):
                        return False, f"⛔ DANGEROUS FOOD: {food} for hypertensive patient", 2

            for allergen in self.safety_rules['allergy_triggers']:
                if allergen.lower() in rec_lower and allergen.lower() in patient_info.get('allergies', []):
                    return False, f"🚨 ALLERGY ALERT: Contains {allergen}", 4

        return True, "✅ Domain-specific safety check passed", 0

    def _generate_safety_message(self, issues: List[str], risk_level: int, confidence: float) -> str:
        """Generate safety message with confidence consideration"""
        if not issues:
            if confidence >= 0.8:
                return "✅ All safety checks passed. High confidence recommendation."
            elif confidence >= 0.6:
                return "✅ Safety checks passed. Moderate confidence - consider review."
            else:
                return "⚠️ Safety checks passed but low confidence. Expert review recommended."

        risk_descriptions = {
            0: "✅ Safe",
            1: "⚠️ Caution advised",
            2: "⚠️ Warning - review recommended",
            3: "🚨 High risk - expert consultation needed",
            4: "⛔ Critical risk - do not proceed"
        }

        header = risk_descriptions.get(risk_level, "Unknown risk level")
        body = " | ".join(issues)
        conf_note = f" (Confidence: {confidence:.1%})"

        return f"{header}: {body}{conf_note}"

# ==============================================
# 🚀 ENHANCED INFERENCE ENGINE
# ==============================================
class EnhancedInferenceEngine:
    """Enhanced inference engine with confidence scoring"""

    def __init__(self, model, tokenizer, safety_system, uhai_instance):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_system = safety_system
        self.uhai = uhai_instance

    def intelligent_query_routing(self, query: str) -> str:
        """Route queries to appropriate domain"""
        clinical_keywords = [
            'patient', 'symptoms', 'diagnosis', 'treatment', 'medication',
            'prescription', 'disease', 'condition', 'medical', 'clinical'
        ]
        
        nutrition_keywords = [
            'meal', 'food', 'nutrition', 'diet', 'eating', 'calories',
            'protein', 'carbs', 'fat', 'vitamins', 'nutrients'
        ]

        query_lower = query.lower()
        clinical_score = sum(1 for keyword in clinical_keywords if keyword in query_lower)
        nutrition_score = sum(1 for keyword in nutrition_keywords if keyword in query_lower)

        if clinical_score > nutrition_score:
            return "clinical"
        elif nutrition_score > clinical_score:
            return "nutrition"
        else:
            return "general"

    def generate_with_confidence(self, 
                                 prompt: str, 
                                 generation_params: Dict = None) -> Tuple[str, Dict]:
        """Generate response with confidence tracking"""
        
        if generation_params is None:
            generation_params = {
                'max_new_tokens': 150,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True,
                'return_dict_in_generate': True,
                'output_scores': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate with detailed outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs.to(self.model.device), # Ensure inputs are on the same device as the model
                **generation_params
            )
        
        # Extract generated text
        # Check if outputs.sequences is not empty before accessing
        if outputs.sequences.numel() > 0:
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            generated_text = "" # No text generated

        # Prepare model outputs for confidence calculation
        model_outputs = {
            'generated_text': generated_text,
            'sequences_scores': getattr(outputs, 'sequences_scores', None),
            'scores': getattr(outputs, 'scores', None)
        }
        
        # Add logits if available
        if hasattr(outputs, 'scores') and outputs.scores:
            # Stack the scores (logits) from each generated token
            # Ensure all scores are on the same device
            model_outputs['logits'] = torch.stack([s.to(self.model.device) for s in outputs.scores], dim=1)
        
        return generated_text, model_outputs

    def infer_with_confidence(self, 
                              query: str, 
                              patient_info: Dict = None) -> Dict:
        """Enhanced inference with comprehensive confidence scoring"""
        
        # Route query
        domain = self.intelligent_query_routing(query)
        
        # Get appropriate context
        if domain == "clinical":
            context = self.uhai._get_clinical_context('general')
            prompt = f"CLINICAL CONTEXT: {context}\n\nPATIENT SCENARIO: {query}\n\nRECOMMENDATION:"
        elif domain == "nutrition":
            context = self.uhai._get_nutrition_context('US')
            prompt = f"NUTRITION CONTEXT: {context}\n\nMEAL DESCRIPTION: {query}\n\nANALYSIS:"
        else:
            context = self.uhai._get_clinical_context('general')
            prompt = f"HEALTHCARE CONTEXT: {context}\n\nSCENARIO: {query}\n\nRECOMMENDATION:"
        
        # Generate response with confidence tracking
        generation_params = {
            'max_new_tokens': 150,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'return_dict_in_generate': True,
            'output_scores': True,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        
        generated_response, model_outputs = self.generate_with_confidence(
            prompt, generation_params
        )
        
        # Initial safety check (without confidence)
        is_safe, safety_message, risk_level = self.safety_system.comprehensive_safety_check(
            generated_response, domain, 0.5, patient_info  # Use placeholder confidence
        )
        
        safety_assessment = {
            'is_safe': is_safe,
            'message': safety_message,
            'risk_level': risk_level
        }
        
        # Calculate comprehensive confidence
        confidence_results = self.uhai.confidence_scorer.calculate_overall_confidence(
            model_outputs=model_outputs,
            response=generated_response,
            query=query,
            domain=domain,
            safety_assessment=safety_assessment,
            patient_info=patient_info,
            generation_params=generation_params
        )
        
        # Re-run safety check with actual confidence
        is_safe, safety_message, risk_level = self.safety_system.comprehensive_safety_check(
            generated_response, domain, confidence_results['overall_confidence'], patient_info
        )
        
        # Update safety assessment
        safety_assessment.update({
            'is_safe': is_safe,
            'message': safety_message,
            'risk_level': risk_level
        })
        
        return {
            "query": query,
            "domain": domain,
            "generated_response": generated_response,
            "confidence_scoring": confidence_results,
            "safety_assessment": safety_assessment,
            "recommendations": self._generate_recommendations(confidence_results, safety_assessment)
        }
    
    def _generate_recommendations(self, confidence_results: Dict, safety_assessment: Dict) -> List[str]:
        """Generate actionable recommendations based on confidence and safety."""
        recommendations = []
        
        overall_confidence = confidence_results['overall_confidence']
        confidence_level = confidence_results['confidence_level']
        safety_risk_level = safety_assessment['risk_level']
        
        if safety_risk_level == 4:
            recommendations.append("⛔ **CRITICAL SAFETY ALERT**: The recommendation poses a critical risk. DO NOT proceed. Immediate expert consultation is required.")
        elif safety_risk_level == 3:
            recommendations.append("🚨 **HIGH RISK**: The recommendation carries a high risk. Expert consultation is strongly needed before any action.")
        elif safety_risk_level == 2:
            recommendations.append("⚠️ **WARNING**: Review of the recommendation by a healthcare professional is strongly recommended due to potential risks.")
        elif safety_risk_level == 1:
            recommendations.append("💡 **CAUTION**: The recommendation has some minor concerns. A quick review by a professional is advised.")
        else: # risk_level == 0 (safe)
            if overall_confidence >= 0.8:
                recommendations.append("✅ **Highly Confident**: This recommendation is generally reliable. However, always cross-reference with a healthcare professional for personalized advice.")
            elif overall_confidence >= 0.6:
                recommendations.append("👍 **Moderately Confident**: This recommendation appears sound, but consulting a healthcare professional for verification is a good practice.")
            else:
                recommendations.append("❓ **Low Confidence**: While no immediate safety issues were detected, the low confidence score suggests this recommendation should be thoroughly reviewed by a healthcare professional.")

        # Add more specific recommendations based on uncertainty types
        uncertainty = confidence_results['uncertainty_quantification']
        if uncertainty['aleatoric_uncertainty'] > 0.4: # Arbitrary threshold for high aleatoric uncertainty
            recommendations.append("🔍 **Data Uncertainty**: The model encountered variability or ambiguity in the input data, leading to higher aleatoric uncertainty. Providing more specific and clear patient information could improve future results.")
        
        if uncertainty['epistemic_uncertainty'] > 0.4: # Arbitrary threshold for high epistemic uncertainty
            recommendations.append("🧠 **Model Uncertainty**: The model's knowledge base might be insufficient or the query falls into a less explored area. Further training or specialized models might be needed for similar queries.")
        
        # Add general disclaimer
        recommendations.append("\n**Important Disclaimer**: This AI system is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any medical concerns.")

        return recommendations

# ==============================================
# 🧪 EXAMPLE USAGE
# ==============================================
if __name__ == "__main__":
    # Initialize the Unified Healthcare AI System
    uhai_system = EnhancedUnifiedHealthcareAI(model_name="distilgpt2")
    
    # Load the model (ensure you have the model downloaded or accessible)
    if not uhai_system.load_model():
        print("Failed to load model. Exiting.")
        exit()

    # Initialize the Enhanced Safety System with loaded rules
    safety_system = EnhancedSafetySystem(uhai_system.safety_rules)

    # Initialize the Enhanced Inference Engine
    inference_engine = EnhancedInferenceEngine(
        uhai_system.model, uhai_system.tokenizer, safety_system, uhai_system
    )

    print("\n--- Running Example Inferences ---")

    # Example 1: Clinical Query (High Confidence Expected)
    clinical_query = "What is the recommended first-line treatment for Type 2 Diabetes, and what lifestyle changes are important?"
    patient_info_diabetes = {
        'age': 55,
        'gender': 'male',
        'conditions': ['diabetes'],
        'medications': [],
        'allergies': []
    }
    print(f"\nQUERY: {clinical_query}")
    result_clinical = inference_engine.infer_with_confidence(clinical_query, patient_info_diabetes)
    print(f"DOMAIN: {result_clinical['domain']}")
    print(f"GENERATED RESPONSE: {result_clinical['generated_response']}")
    print(f"CONFIDENCE: {result_clinical['confidence_scoring']['overall_confidence']:.2f} ({result_clinical['confidence_scoring']['confidence_level']})")
    print(f"SAFETY ASSESSMENT: {result_clinical['safety_assessment']['message']} (Risk Level: {result_clinical['safety_assessment']['risk_level']})")
    print("RECOMMENDATIONS:")
    for rec in result_clinical['recommendations']:
        print(f"- {rec}")
    print(f"DETAILED METRICS: {json.dumps(result_clinical['confidence_scoring']['detailed_metrics'], indent=2)}")
    print(f"UNCERTAINTY: {json.dumps(result_clinical['confidence_scoring']['uncertainty_quantification'], indent=2)}")
    print("-" * 30)

    # Example 2: Nutrition Query (Potentially Moderate Confidence)
    nutrition_query = "Suggest a healthy dinner meal plan for someone looking to reduce calorie intake, based on US nutrition standards."
    patient_info_nutrition = {
        'age': 30,
        'gender': 'female',
        'conditions': [],
        'medications': [],
        'allergies': ['dairy']
    }
    print(f"\nQUERY: {nutrition_query}")
    result_nutrition = inference_engine.infer_with_confidence(nutrition_query, patient_info_nutrition)
    print(f"DOMAIN: {result_nutrition['domain']}")
    print(f"GENERATED RESPONSE: {result_nutrition['generated_response']}")
    print(f"CONFIDENCE: {result_nutrition['confidence_scoring']['overall_confidence']:.2f} ({result_nutrition['confidence_scoring']['confidence_level']})")
    print(f"SAFETY ASSESSMENT: {result_nutrition['safety_assessment']['message']} (Risk Level: {result_nutrition['safety_assessment']['risk_level']})")
    print("RECOMMENDATIONS:")
    for rec in result_nutrition['recommendations']:
        print(f"- {rec}")
    print(f"DETAILED METRICS: {json.dumps(result_nutrition['confidence_scoring']['detailed_metrics'], indent=2)}")
    print(f"UNCERTAINTY: {json_nutrition['confidence_scoring']['uncertainty_quantification'], indent=2}")
    print("-" * 30)

    # Example 3: Query with potential safety concern (dangerous drug)
    safety_query_drug = "Is 'thalidomide' a good medication for sleep?"
    print(f"\nQUERY: {safety_query_drug}")
    result_safety_drug = inference_engine.infer_with_confidence(safety_query_drug)
    print(f"DOMAIN: {result_safety_drug['domain']}")
    print(f"GENERATED RESPONSE: {result_safety_drug['generated_response']}")
    print(f"CONFIDENCE: {result_safety_drug['confidence_scoring']['overall_confidence']:.2f} ({result_safety_drug['confidence_scoring']['confidence_level']})")
    print(f"SAFETY ASSESSMENT: {result_safety_drug['safety_assessment']['message']} (Risk Level: {result_safety_drug['safety_assessment']['risk_level']})")
    print("RECOMMENDATIONS:")
    for rec in result_safety_drug['recommendations']:
        print(f"- {rec}")
    print("-" * 30)

    # Example 4: Query with patient-specific safety concern (allergy)
    safety_query_allergy = "Suggest a protein source for someone with a peanut allergy."
    patient_info_allergy = {
        'age': 25,
        'gender': 'female',
        'conditions': [],
        'medications': [],
        'allergies': ['peanuts']
    }
    print(f"\nQUERY: {safety_query_allergy}")
    result_safety_allergy = inference_engine.infer_with_confidence(safety_query_allergy, patient_info_allergy)
    print(f"DOMAIN: {result_safety_allergy['domain']}")
    print(f"GENERATED RESPONSE: {result_safety_allergy['generated_response']}")
    print(f"CONFIDENCE: {result_safety_allergy['confidence_scoring']['overall_confidence']:.2f} ({result_safety_allergy['confidence_scoring']['confidence_level']})")
    print(f"SAFETY ASSESSMENT: {result_safety_allergy['safety_assessment']['message']} (Risk Level: {result_safety_allergy['safety_assessment']['risk_level']})")
    print("RECOMMENDATIONS:")
    for rec in result_safety_allergy['recommendations']:
        print(f"- {rec}")
    print("-" * 30)

    # Example 5: General health query (moderate confidence, as specific medical context might be limited)
    general_query = "What are some general tips for improving gut health?"
    print(f"\nQUERY: {general_query}")
    result_general = inference_engine.infer_with_confidence(general_query)
    print(f"DOMAIN: {result_general['domain']}")
    print(f"GENERATED RESPONSE: {result_general['generated_response']}")
    print(f"CONFIDENCE: {result_general['confidence_scoring']['overall_confidence']:.2f} ({result_general['confidence_scoring']['confidence_level']})")
    print(f"SAFETY ASSESSMENT: {result_general['safety_assessment']['message']} (Risk Level: {result_general['safety_assessment']['risk_level']})")
    print("RECOMMENDATIONS:")
    for rec in result_general['recommendations']:
        print(f"- {rec}")
    print("-" * 30)