# ==============================================
# 🏥 COMPLETE UNIFIED HEALTHCARE AI SYSTEM
# Clinical Decision Support + Nutritional Analysis + Confidence Scoring
# ==============================================

import os
import json
import torch
import platform
import torch.backends.mps
import google.generativeai as genai # Keep this for genai.configure and potential future use, though Part is moved
from google.cloud import aiplatform
from google.api_core.client_options import ClientOptions
import vertexai


import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import scipy.stats as stats
from scipy.special import softmax
from vertexai.generative_models import GenerativeModel,Content, Part, HarmCategory, HarmBlockThreshold, SafetySetting
import pandas as pd
import numpy as np
import base64
import requests
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import scipy.stats as stats
from scipy.special import softmax
import math

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. Visualizations will be skipped.")
SAFETY_RULES = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]
# ML/AI Libraries
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        pipeline
    )
    from datasets import Dataset, DatasetDict, load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
    print("transformer haii hyayyy")
except ImportError:
    print("⚠️ Transformers not available. Using fallback mode.")
    TRANSFORMERS_AVAILABLE = False

# Visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Visualization features disabled.")
    PLOTLY_AVAILABLE = False

# Environment setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not available. Environment variables should be set manually.")

# ==============================================
# 🎯 CONFIDENCE SCORING SYSTEM
# ==============================================

class ConfidenceScorer:
    """
    Confidence scoring system for healthcare AI recommendations
    """

    def __init__(self):
        self.confidence_weights = {
            'model_confidence': 0.25,
            'response_coherence': 0.25,
            'medical_validity': 0.25,
            'safety_score': 0.25
        }

        # Medical entity patterns for validation
        self.medical_patterns = {
            'medications': r'\b(?:mg|mcg|ml|tablet|capsule|injection|daily|twice|thrice|medication|drug|prescription)\b',
            'dosages': r'\b\d+\s*(?:mg|mcg|ml|g|kg|units?)\b',
            'conditions': r'\b(?:diabetes|hypertension|cancer|infection|pain|fever|disease|condition|illness)\b',
            'lab_values': r'\b\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|%|bpm)\b',
            'nutrition_terms': r'\b(?:calories|protein|carbohydrate|fat|vitamin|nutrient|diet|healthy|balanced|portion|nutrition)\b'
        }

        self.calibration_params = {
            'temperature': 1.0, # Adjust these if needed for calibration
            'bias': 0.0
        }

    def calculate_model_confidence(self, model_outputs: Dict, generation_params: Dict) -> Tuple[float, Dict]:
        """
        Calculate model's internal confidence for OpenAI API responses.
        Direct sequence/logit analysis is not possible.
        Approximations are used for entropy and probability confidence.
        """
        confidence_metrics = {}

        # --- Approximation for Entropy/Probability Confidence (Not directly available from OpenAI API) ---
        # Since 'sequences' and direct log probabilities are not exposed by OpenAI API,
        # we'll use a heuristic. High temperature/low top_p can indicate higher model uncertainty.
        # Conversely, a well-formed, coherent response might suggest higher internal confidence.
        
        # Heuristic: A longer, more detailed response might imply the model was "more confident"
        # in generating comprehensive content, up to a reasonable limit.
        response_text = model_outputs.get('generated_text', '')
        response_length = len(response_text.split())

        # Base confidence based on response length heuristic
        if response_length > 100:
            heuristic_base_confidence = 0.8
        elif response_length > 50:
            heuristic_base_confidence = 0.7
        elif response_length > 10:
            heuristic_base_confidence = 0.6
        else: # Very short response
            heuristic_base_confidence = 0.4
        
        # Adjust based on temperature and top_p from generation parameters
        temperature = generation_params.get('temperature', 0.7)
        top_p = generation_params.get('top_p', 0.9)

        # Penalize higher temperatures (more randomness -> less "confident" deterministic output)
        temp_factor = max(0.0, 1.0 - (temperature / 1.5)) # Scales from ~1.0 (temp=0) to ~0.3 (temp=1)
        
        # Reward higher top_p (more focused sampling -> potentially more "confident")
        top_p_factor = top_p # Direct use of top_p value

        # Combine heuristic and parameter factors for "entropy" and "probability"
        # These are now approximations, not true entropy/probability from logits
        confidence_metrics['entropy_confidence'] = heuristic_base_confidence * temp_factor
        confidence_metrics['probability_confidence'] = heuristic_base_confidence * top_p_factor

        # Ensure values are within [0, 1]
        confidence_metrics['entropy_confidence'] = min(max(confidence_metrics['entropy_confidence'], 0.1), 0.95)
        confidence_metrics['probability_confidence'] = min(max(confidence_metrics['probability_confidence'], 0.1), 0.95)

        # --- Parameter Confidence (already robust) ---
        temp_penalty = max(0.1, 1.0 - temperature) # Use direct temperature here
        top_p_boost = top_p
        param_confidence = (temp_penalty + top_p_boost) / 2
        confidence_metrics['parameter_confidence'] = param_confidence
        
        # --- Length Confidence (already robust) ---
        if 10 <= response_length <= 100:
            length_confidence = 0.9
        elif 5 <= response_length < 10:
            length_confidence = 0.7
        elif response_length < 5:
            length_confidence = 0.3
        elif response_length > 200:
            length_confidence = 0.6
        else:
            length_confidence = 0.8
        confidence_metrics['length_confidence'] = length_confidence

        # --- Overall Model Confidence ---
        model_confidence = np.mean([
            confidence_metrics['entropy_confidence'],
            confidence_metrics['probability_confidence'],
            confidence_metrics['parameter_confidence'],
            confidence_metrics['length_confidence']
        ])+0.01

        return model_confidence, confidence_metrics

    def calculate_response_coherence(self, response: str, query: str) -> Tuple[float, Dict]:
        """Calculate response coherence and relevance"""
        coherence_metrics = {}

        if not response or not response.strip():
            return 0.1, {'error': 'Empty response'}

        response_words = response.split()
        if len(response_words) < 5:
            completeness = 0.2
        elif len(response_words) < 15:
            completeness = 0.6
        elif len(response_words) <= 50:
            completeness = 1.0
        else:
            completeness = 0.8
        coherence_metrics['completeness'] = completeness

        unique_words = set(response_words)
        if len(response_words) > 0:
            repetition_ratio = len(unique_words) / len(response_words)
        else:
            repetition_ratio = 0
        
        repetitive_phrases = re.findall(r'(\b\w+\b)(?:\s+\1){2,}', response.lower())
        if repetitive_phrases:
            repetition_ratio *= 0.3 # Penalize if repetitive phrases are found
        
        coherence_metrics['repetition_score'] = repetition_ratio

        medical_terms = 0
        for pattern in self.medical_patterns.values():
            medical_terms += len(re.findall(pattern, response.lower()))

        medical_density = min(medical_terms / max(len(response_words), 1) * 10, 1.0) # Scaled density
        coherence_metrics['medical_terminology'] = medical_density

        query_words = set(query.lower().split())
        response_words_set = set(response.lower().split())
        
        if len(query_words) > 0:
            relevance = len(query_words.intersection(response_words_set)) / len(query_words)
        else:
            relevance = 0
        coherence_metrics['relevance'] = relevance

        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        if len(sentences) > 0:
            structural_score = len(valid_sentences) / len(sentences)
        else:
            structural_score = 0
        coherence_metrics['structural_coherence'] = structural_score

        coherence_score = (
            completeness * 0.25 +
            repetition_ratio * 0.30 +
            medical_density * 0.20 +
            relevance * 0.15 +
            structural_score * 0.10
        )

        return coherence_score, coherence_metrics

    def calculate_medical_validity(self, response: str, domain: str, patient_info: Dict = None) -> Tuple[float, Dict]:
        """Calculate medical validity score"""
        validity_metrics = {}

        if not response or not response.strip():
            return 0.1, {'error': 'Empty response'}

        response_lower = response.lower()

        if domain == "clinical":
            clinical_terms = [
                'treatment', 'medication', 'diagnosis', 'symptoms', 'patient',
                'therapy', 'prescription', 'dosage', 'side effect', 'contraindication',
                'clinical', 'medical', 'condition', 'disease'
            ]
            term_presence = sum(1 for term in clinical_terms if term in response_lower)
            terminology_score = min(term_presence / 5, 1.0)
        elif domain == "nutrition":
            nutrition_terms = [
                'calories', 'protein', 'carbohydrate', 'fat', 'vitamin',
                'nutrient', 'diet', 'healthy', 'balanced', 'portion',
                'nutrition', 'food', 'meal', 'eating'
            ]
            term_presence = sum(1 for term in nutrition_terms if term in response_lower)
            terminology_score = min(term_presence / 5, 1.0)
        else:
            terminology_score = 0.5

        validity_metrics['terminology_score'] = terminology_score

        dosages = re.findall(self.medical_patterns['dosages'], response)
        valid_dosages = 0
        for dosage in dosages:
            numbers = re.findall(r'\d+(?:\.\d+)?', dosage)
            if numbers:
                try:
                    value = float(numbers[0])
                    # Example basic range validation for dosages
                    if 0.001 <= value <= 5000: # Adjust range as appropriate
                        valid_dosages += 1
                except ValueError:
                    continue

        dosage_validity = valid_dosages / max(len(dosages), 1) if dosages else 1.0
        validity_metrics['dosage_validity'] = dosage_validity

        medical_flow_patterns = [
            r'(?:diagnos|assess|evaluat)\w*.*(?:treat|manag|therap)',
            r'(?:symptom|sign)\w*.*(?:medication|treatment|therapy)',
            r'(?:condition|disease)\w*.*(?:management|treatment|care)',
            r'(?:first.?line|initial).*(?:second.?line|alternative|next)',
            r'(?:recommend|suggest|advise).*(?:consider|monitor|follow)'
        ]

        flow_score = 0
        for pattern in medical_flow_patterns:
            if re.search(pattern, response_lower):
                flow_score += 1

        logical_flow = min(flow_score / 3, 1.0) # Max 3 matched patterns for full score
        validity_metrics['logical_flow'] = logical_flow

        if patient_info:
            appropriateness_score = self._assess_patient_appropriateness(response, patient_info)
        else:
            appropriateness_score = 0.8 # Default if no patient info

        validity_metrics['patient_appropriateness'] = appropriateness_score

        medical_validity = (
            terminology_score * 0.35 +
            dosage_validity * 0.20 +
            logical_flow * 0.25 +
            appropriateness_score * 0.20
        )

        return medical_validity, validity_metrics

    def _assess_patient_appropriateness(self, response: str, patient_info: Dict) -> float:
        """Assess if response is appropriate for patient demographics"""
        score = 1.0
        response_lower = response.lower()

        age = patient_info.get('age', 0)
        if age > 65:
            # Examples of risky recommendations for elderly
            risky_for_elderly = ['benzodiazepine', 'anticholinergic', 'high dose', 'sedative']
            for risk in risky_for_elderly:
                if risk in response_lower:
                    score -= 0.2

        if patient_info.get('pregnant', False):
            # Examples of pregnancy risks
            pregnancy_risks = ['warfarin', 'ace inhibitor', 'high mercury', 'isotretinoin', 'certain vaccines']
            for risk in pregnancy_risks:
                if risk in response_lower:
                    score -= 0.3

        conditions = patient_info.get('conditions', [])
        # Simple checks for contraindications based on conditions
        if 'diabetes' in conditions and ('high sugar' in response_lower or 'simple carbohydrates' in response_lower):
            score -= 0.25
        if 'hypertension' in conditions and ('high sodium' in response_lower or 'excess salt' in response_lower):
            score -= 0.25
        if 'kidney disease' in conditions and ('high potassium' in response_lower or 'creatine supplements' in response_lower):
            score -= 0.25
        if 'liver disease' in conditions and 'alcohol' in response_lower:
            score -= 0.25


        return max(score, 0.0) # Score cannot be negative

    def calculate_overall_confidence(self, model_outputs: Dict, response: str, query: str, 
                                   domain: str, safety_assessment: Dict, 
                                   patient_info: Dict = None, generation_params: Dict = None) -> Dict:
        """Calculate overall confidence score"""
        if generation_params is None:
            # Provide default generation params if not supplied, crucial for calculate_model_confidence
            generation_params = {'temperature': 0.6, 'top_p': 0.9}

        if 'generated_text' not in model_outputs and response:
            model_outputs['generated_text'] = response # Ensure generated_text is present for model_confidence

        try:
            model_conf, model_metrics = self.calculate_model_confidence(model_outputs, generation_params)
            coherence_conf, coherence_metrics = self.calculate_response_coherence(response, query)
            medical_conf, medical_metrics = self.calculate_medical_validity(response, domain, patient_info)
            
            # safety_assessment should contain 'risk_level' from EnhancedSafetySystem
            # Risk levels are 0-4 (safe to critical), so scale to 0-1 for confidence score
            safety_score = 1.0 - (safety_assessment.get('risk_level', 0) / 4.0) # Max risk_level is 4

            overall_confidence = (
                model_conf * self.confidence_weights['model_confidence'] +
                coherence_conf * self.confidence_weights['response_coherence'] +
                medical_conf * self.confidence_weights['medical_validity'] +
                safety_score * self.confidence_weights['safety_score']
            )

            calibrated_confidence = self._calibrate_confidence(overall_confidence)
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
                'uncertainty_quantification': self._quantify_uncertainty(calibrated_confidence, model_metrics)
            }
        except Exception as e:
            return {
                'overall_confidence': 0.1,
                'confidence_level': "Very Low",
                'error': str(e),
                'component_scores': {
                    'model_confidence': 0.1,
                    'response_coherence': 0.1,
                    'medical_validity': 0.1,
                    'safety_score': 1.0 # If error, assume safety score is high (no specific dangerous content detected by scorer)
                },
                'detailed_metrics': {},
                'uncertainty_quantification': {}
            }

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score using a sigmoid-like function"""
        # Ensure calibration_params are float
        temp = float(self.calibration_params.get('temperature', 1.0))
        bias = float(self.calibration_params.get('bias', 0.0))
        
        # Avoid division by zero if temperature is too small
        if temp == 0:
            return 1.0 if raw_confidence >= bias else 0.0

        calibrated = 1 / (1 + np.exp(-(raw_confidence - bias) / temp))
        return max(0.0, min(1.0, calibrated))

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
        # These will now be based on the approximated model_metrics
        entropy_conf = model_metrics.get('entropy_confidence', 0.5)
        aleatoric_uncertainty = 1.0 - entropy_conf

        prob_conf = model_metrics.get('probability_confidence', 0.5)
        epistemic_uncertainty = 1.0 - prob_conf

        total_uncertainty = 1.0 - confidence

        return {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence_interval': {
                'lower_bound': max(0.0, confidence - total_uncertainty/2),
                'upper_bound': min(1.0, confidence + total_uncertainty/2)
            }
        }
# ==============================================
# 🔒 ENHANCED SAFETY SYSTEM
# ==============================================

class EnhancedSafetySystem:
    """Enhanced safety system with confidence-aware checks"""

    def __init__(self, safety_rules: Dict = None): # Initialize with SAFETY_RULES
        if safety_rules is None:
            self.safety_rules = self._default_safety_rules()
        else:
            self.safety_rules = safety_rules
        self.risk_levels = {
            'safe': 0,
            'caution': 1,
            'warning': 2,
            'danger': 3,
            'critical': 4
        }

    def _default_safety_rules(self) -> Dict:
        """Load comprehensive safety rules"""
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
            ],
            'sensitive_topics': [ # Added sensitive topics to safety_rules
                'suicide', 'self-harm', 'child abuse', 'illegal activities'
            ]
        }

    def comprehensive_safety_check(self, recommendation: str, domain: str, 
                                 confidence_score: float, patient_info: Dict = None) -> Dict: # Return Dict
        """Enhanced safety check considering confidence"""
        if not recommendation or not recommendation.strip():
            return {
                "is_safe": False, 
                "message": "⚠️ Empty or invalid recommendation", 
                "risk_level": self.risk_levels['warning'],
                "overall_safety_score": 0.5 # Default score for invalid
            }

        issues = []
        max_risk_level = 0

        # Adjust risk multiplier based on confidence score
        if confidence_score < 0.5:
            risk_multiplier = 1.5
        elif confidence_score < 0.3:
            risk_multiplier = 2.0
        else:
            risk_multiplier = 1.0

        # Apply basic safety checks
        basic_safe, basic_msg, basic_risk = self._basic_safety_check(recommendation)
        if not basic_safe:
            issues.append(basic_msg)
            max_risk_level = max(max_risk_level, int(basic_risk * risk_multiplier))

        # Apply domain-specific safety checks
        domain_safe, domain_msg, domain_risk = self._domain_specific_safety_check(
            recommendation, domain, patient_info
        )
        if not domain_safe:
            issues.append(domain_msg)
            max_risk_level = max(max_risk_level, int(domain_risk * risk_multiplier))

        # Check content quality
        repetition_safe, rep_msg, rep_risk = self._check_content_quality(recommendation)
        if not repetition_safe:
            issues.append(rep_msg)
            max_risk_level = max(max_risk_level, rep_risk)
            
        # Ensure max_risk_level does not exceed the defined critical level
        max_risk_level = min(max_risk_level, self.risk_levels['critical'])

        is_safe = max_risk_level <= self.risk_levels['caution'] # "Safe" if risk is caution or lower
        safety_message = self._generate_safety_message(issues, max_risk_level, confidence_score)
        
        # Calculate overall safety score (e.g., 1.0 for safe, 0.0 for critical)
        overall_safety_score = 1.0 - (max_risk_level / self.risk_levels['critical'])


        return {
            "is_safe": is_safe,
            "message": safety_message,
            "risk_level": max_risk_level,
            "safety_level": self._get_safety_level(max_risk_level), # Add categorical level
            "overall_safety_score": overall_safety_score # For confidence scorer
        }

    def _basic_safety_check(self, recommendation: str) -> Tuple[bool, str, int]:
        """Basic safety checks"""
        rec_lower = recommendation.lower()

        for drug in self.safety_rules.get('dangerous_drugs', []):
            if drug.lower() in rec_lower:
                return False, f"⛔ DANGEROUS DRUG: {drug} is contraindicated", self.risk_levels['critical']

        for combo in self.safety_rules.get('contraindicated_combinations', []):
            if all(term.strip().lower() in rec_lower for term in combo.split('+')):
                return False, f"⛔ CONTRAINDICATED COMBINATION: {combo} detected", self.risk_levels['danger']

        # Check for sensitive topics / harmful content
        for topic in self.safety_rules.get('sensitive_topics', []):
            if topic.lower() in rec_lower:
                return False, f"⚠️ SENSITIVE TOPIC: Content related to '{topic}' detected. Requires careful review.", self.risk_levels['critical']


        return True, "✅ Basic safety check passed", self.risk_levels['safe']

    def _domain_specific_safety_check(self, recommendation: str, domain: str, patient_info: Dict) -> Tuple[bool, str, int]:
        """Perform domain-specific safety checks"""
        rec_lower = recommendation.lower()

        if domain == "nutrition" and patient_info:
            for food in self.safety_rules.get('dangerous_foods', []):
                # This logic is more complex and depends on how you define 'dangerous_foods' in your rules
                # The provided example assumes `food` string itself contains patient-specific flags.
                # A more robust system would have separate rule objects with conditions.
                
                # Simplified example:
                if 'immunocompromised' in food.lower() and patient_info.get('immunocompromised', False) and food.lower().split('(')[0].strip() in rec_lower:
                    return False, f"⛔ DANGEROUS FOOD: {food} for immunocompromised patient", self.risk_levels['danger']
                if 'pregnant women' in food.lower() and patient_info.get('pregnant', False) and food.lower().split('(')[0].strip() in rec_lower:
                    return False, f"⛔ DANGEROUS FOOD: {food} for pregnant patient", self.risk_levels['danger']
                if 'hypertensive patients' in food.lower() and 'hypertension' in patient_info.get('conditions', []) and food.lower().split('(')[0].strip() in rec_lower:
                    return False, f"⛔ DANGEROUS FOOD: {food} for hypertensive patient", self.risk_levels['danger']
                
        # You might also add specific clinical domain checks here, e.g., if a drug dosage is abnormally high/low

        return True, "✅ Domain-specific safety check passed", self.risk_levels['safe']

    def _check_content_quality(self, recommendation: str) -> Tuple[bool, str, int]:
        """Check for repetitive or nonsensical content"""
        words = recommendation.split()
        if len(words) < 3: # Too short
            return False, "⚠️ Response too short for meaningful assessment", self.risk_levels['caution']

        # Check for excessive repetition (more robust)
        # Using n-grams to detect repeating phrases, not just single words
        if len(words) > 0:
            from collections import Counter
            word_counts = Counter(words)
            most_common_word_freq = word_counts.most_common(1)[0][1] if word_counts else 0
            if most_common_word_freq > len(words) * 0.3: # If any single word repeats too much
                 return False, "⚠️ Excessive word repetition detected", self.risk_levels['warning']

            # Check for repeating short phrases (e.g., "I am an AI. I am an AI.")
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            if bigram_counts and bigram_counts.most_common(1)[0][1] > len(bigrams) * 0.2:
                 return False, "⚠️ Excessive phrase repetition detected", self.risk_levels['warning']
        
        # Add checks for obvious filler/boilerplate that might indicate low quality
        filler_patterns = [r'as an ai model', r'i cannot provide personalized', r'consult a healthcare professional']
        for pattern in filler_patterns:
            if re.search(pattern, recommendation.lower()):
                # If too much filler, it might be low quality but not necessarily dangerous
                pass # Can add a penalty if you want
        
        return True, "✅ Content quality check passed", self.risk_levels['safe']

    def _generate_safety_message(self, issues: List[str], risk_level: int, confidence_score: float) -> str:
        """Generate comprehensive safety message"""
        if not issues:
            return f"✅ All safety checks passed (Confidence: {confidence_score:.1%})"

        message = f"⚠️ Safety concerns detected (Risk Level: {risk_level}/4 - {self._get_safety_level(risk_level)}):\n"
        for issue in issues:
            message += f"• {issue}\n"
        
        if confidence_score < 0.6: # If confidence is below moderate
            message += f"• Low confidence score ({confidence_score:.1%}) for this recommendation."

        return message.strip()

    def _get_safety_level(self, risk_level: int) -> str:
        """Convert numeric risk level to categorical level"""
        if risk_level == self.risk_levels['safe']:
            return "Safe"
        elif risk_level == self.risk_levels['caution']:
            return "Caution"
        elif risk_level == self.risk_levels['warning']:
            return "Warning"
        elif risk_level == self.risk_levels['danger']:
            return "Danger"
        elif risk_level == self.risk_levels['critical']:
            return "Critical"
        else:
            return "Unknown"

# ==============================================
# 🎨 HEALTHCARE AI VISUALIZER
# ==============================================
class HealthcareAIVisualizer:
    def plot_confidence_breakdown(self, confidence_scores) -> Optional[str]:
        if not PLOTLY_AVAILABLE:
            return None
        # Placeholder for Plotly visualization
        fig = go.Figure(data=[go.Bar(y=['Confidence'], x=[confidence_scores.get('overall_confidence', 0.0)])])
        return fig.to_json() # Return as JSON string for embedding

    def generate_safety_report(self, safety_evaluation) -> Optional[str]:
        if not PLOTLY_AVAILABLE:
            return None
        # Placeholder for Plotly visualization
        fig = go.Figure(data=[go.Indicator(mode = "gauge+number", value = 1 if safety_evaluation.get('is_safe', False) else 0,
                                           domain = {'x': [0, 1], 'y': [0, 1]},
                                           title = {'text': "Safety Status"})])
        return fig.to_json() # Return as JSON string for embedding

# ==============================================
# 🧠 UNIFIED HEALTHCARE AI MODEL
# ==============================================

class HealthcareAISystem:
    def __init__(self,model_name: str = None, tuned_model_id: str = None):
        # ... (other initializations)
        # Initialize Vertex AI
        project_id = os.getenv("GCP_PROJECT_ID")
        location = "us-central1"
        
        if not project_id or not location:
            print("GCP_PROJECT_ID or GCP_LOCATION environment variables are not set. Vertex AI models may not load.")
            self.vertexai_initialized = False
        else:
            try:
                vertexai.init(project=project_id, location=location)
                self.vertexai_initialized = True
                print(f"Vertex AI initialized for project {project_id} in {location}")
            except Exception as e:
                print(f"Failed to initialize Vertex AI: {e}")
                self.vertexai_initialized = False

        self.vertexai_model = None
        self.tuned_model_id = os.getenv("EN_ID")
        self.model_name ="gemini-2.0-flash-lite-001" 
        self.clinical_model_id = "gemini-2.5-flash" # New attribute for clinical-specific model
        self.clinical_vertexai_model = None
        project_id = os.getenv("GCP_PROJECT_ID")
        location = "us-central1"
        
        if not project_id or not location:
            print("GCP_PROJECT_ID or GCP_LOCATION environment variables are not set. Vertex AI models may not load.")
            self.vertexai_initialized = False
        else:
            try:
                vertexai.init(project=project_id, location=location)
                self.vertexai_initialized = True
                print(f"Vertex AI initialized for project {project_id} in {location}")
            except Exception as e:
                print(f"Failed to initialize Vertex AI: {e}")
                self.vertexai_initialized = False

        self.vertexai_model = None
        self.tuned_model_id = os.getenv("EN_ID")
        self.model_name = "gemini-2.0-flash-lite-001" 
        
        # Initialize ConfidenceScorer and EnhancedSafetySystem
        self.confidence_scorer = ConfidenceScorer()
        self.safety_system = EnhancedSafetySystem() # ADDED: Initialize safety_system
        self.visualizer = HealthcareAIVisualizer() # Initialize visualizer

        self._is_initialized = self.vertexai_initialized # Set _is_initialized based on vertexai initialization
        self._setup_authentication() # Call authentication setup
        # ... (rest of __init__)
    
    def load_model(self):
        if not self.vertexai_initialized:
            print("Vertex AI not initialized. Cannot load models from Vertex AI.")
            return False

        success = True

        # Load general model
        try:
            if self.tuned_model_id:
                print(f"Attempting to load general tuned model: {self.tuned_model_id}")
                self.vertexai_model = GenerativeModel(self.tuned_model_id)
                print(f"Loaded general tuned GenerativeModel: {self.tuned_model_id}")
            elif self.model_name.startswith("gemini"):
                print(f"Attempting to load general base Gemini model: {self.model_name}")
                self.vertexai_model = GenerativeModel(self.model_name)
                print(f"Loaded general base GenerativeModel: {self.model_name}")
            else:
                print(f"Attempting to load general model as Endpoint or TextGenerationModel: {self.model_name}")
                if self.model_name.startswith("text-bison"):
                    self.vertexai_model = vertexai.preview.language_models.TextGenerationModel.from_pretrained(self.model_name)
                    print(f"Loaded general TextGenerationModel: {self.model_name}")
                else:
                    endpoint_name = self.model_name
                    self.vertexai_model = aiplatform.Endpoint(endpoint_name=endpoint_name)
                    print(f"Loaded general aiplatform.Endpoint: {endpoint_name}")
        except Exception as e:
            print(f"Error loading general model from Vertex AI: {e}")
            self.vertexai_model = None
            success = False

        # Load clinical-specific model
        if self.clinical_model_id:
            try:
                print(f"Attempting to load clinical model: {self.clinical_model_id}")
                self.clinical_vertexai_model = GenerativeModel(self.clinical_model_id)
                print(f"Loaded clinical GenerativeModel: {self.clinical_model_id}")
            except Exception as e:
                print(f"Error loading clinical model from Vertex AI: {e}")
                self.clinical_vertexai_model = None
                success = False
        else:
            print("No specific clinical model ID provided. Clinical queries will use the general model.")

        return success    
    @property
    def is_initialized(self):
        return self._is_initialized
        
    @property
    def is_initialized(self):
        """Public property to check initialization status"""
        return self._is_initialized


    def _setup_authentication(self):
        """Setup authentication for various services"""
        auth_status = {
            'huggingface': False,
            'wandb': False,
            'nutritionix': False
        }

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY) # THIS IS THE CORRECT WAY FOR GEMINI
            print("✅ Gemini API key loaded.")
        else:
            print("⚠️ GEMINI_API_KEY environment variable not set. OpenAI API calls will fail.")


    
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

    def intelligent_query_routing(self, query: str) -> str:
        """Route queries to appropriate domain"""
        clinical_keywords = [
            'patient', 'symptoms', 'diagnosis', 'treatment', 'medication',
            'prescription', 'disease', 'condition', 'medical', 'clinical',
            'doctor', 'hospital', 'therapy', 'drug', 'dose', 'side effect'
        ]

        nutrition_keywords = [
            'meal', 'food', 'nutrition', 'diet', 'eating', 'calories',
            'protein', 'carbs', 'fat', 'vitamins', 'nutrients', 'recipe',
            'healthy eating', 'weight', 'fiber', 'minerals'
        ]

        query_lower = query.lower()
        
        clinical_score = sum(1 for word in clinical_keywords if word in query_lower)
        nutrition_score = sum(1 for word in nutrition_keywords if word in query_lower)
        
        if clinical_score > nutrition_score:
            return "clinical"
        elif nutrition_score > clinical_score:
            return "nutrition"
        else:
            if any(term in query_lower for term in ['mg', 'dose', 'rx', 'med', 'treatment']):
                return "clinical"
            elif any(term in query_lower for term in ['calories', 'recipe', 'meal plan']):
                return "nutrition"
            return "nutrition"
            
    def _get_medical_context(self, question: str, domain: str) -> str:
        """Retrieve relevant medical context based on query and domain."""
        guidelines = self._load_medical_guidelines()
        nutrition_standards = self._load_nutrition_standards()

        context = ""
        if domain == "clinical":
            # Search for keywords in guidelines
            for condition, info in guidelines.items():
                if condition.lower() in question.lower():
                    context += f"Medical Guideline for {condition.capitalize()}:\n"
                    for key, value in info.items():
                        context += f"- {key.replace('_', ' ').capitalize()}: {value}\n"
            if not context:
                context = "No specific medical guidelines found for this query. Providing general medical context."
        elif domain == "nutrition":
            # Example: provide general nutrition standards for a default country (e.g., US)
            # In a real system, you'd determine the user's country or ask for it.
            us_standards = nutrition_standards.get("US", {})
            if us_standards:
                context += "General US Nutrition Guidelines (Daily Values):\n"
                for nutrient, value in us_standards.items():
                    context += f"- {nutrient.replace('_', ' ').capitalize()}: {value}\n"
            else:
                context = "No specific nutrition standards found. Providing general nutritional information."
        else: # general_clinical, prescription, diagnosis, safety etc.
            context = "Searching general healthcare knowledge base..."
            # For a real system, you would integrate a more sophisticated RAG (Retrieval Augmented Generation)
            # here, using a vector database to fetch relevant documents based on the `question`.
            # This would involve:
            # 1. Embedding the `question`.
            # 2. Searching a vector database of medical articles, drug databases, etc.
            # 3. Retrieving top-k relevant document snippets.
            # 4. Concatenating them to form the `medical_context`.
        return context

    
    def _construct_prompt(self, question: str, domain: str, patient_info: Dict, medical_context: str) -> List[Content]:
        prompt_parts = []

        system_instruction = (
            "You are a highly knowledgeable and ethical Healthcare AI assistant. "
            "Provide accurate, concise, and helpful information based on current medical understanding. "
            "ALWAYS advise consulting a qualified healthcare professional for diagnosis, treatment, or personalized medical advice. "
            "Do not provide definitive diagnoses or prescribe treatments."
        )
        # Construct Content objects explicitly
        prompt_parts.append(Content(role="user", parts=[Part.from_text(system_instruction)]))
        prompt_parts.append(Content(role="model", parts=[Part.from_text("Understood. I will provide helpful information and emphasize professional consultation.")]))

        patient_info_str = (
            f"Patient Information: Age: {patient_info.get('age', 'N/A')}, Gender: {patient_info.get('gender', 'N/A')}, "
            f"Conditions: {', '.join(patient_info.get('conditions', ['N/A']))}, "
            f"Medications: {', '.join(patient_info.get('medications', ['N/A']))}."
        )

        core_prompt = (
            f"Based on the following context, patient information, and your expertise, think step by step to formulate your response:\n\n"
            f"Medical Context: {medical_context}\n"
            f"Patient Information: {patient_info_str}\n"
            f"Question: {question}\n\n"
            f"First, outline the key considerations or facts relevant to the question. "
            f"Then, based on these considerations, provide your comprehensive and medically sound response. "
            f"Finally, clearly state the key takeaway or recommendation. "
            f"Remember to disclaim that you are an AI and cannot provide personalized medical advice."
        )
        # Construct Content objects explicitly
        prompt_parts.append(Content(role="user", parts=[Part.from_text(core_prompt)]))
        return prompt_parts
    def _determine_domain(self, query: str) -> str:
        """ Determines the clinical domain of the query based on keywords. This is a fallback if no domain is explicitly provided by the frontend. """
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["diet", "food", "nutrition", "meal plan", "calories", "eating"]):
            return "nutrition"
        elif any(keyword in query_lower for keyword in ["medication", "prescription", "drug", "pharmacy", "dose", "tablet"]):
            return "prescription"
        elif any(keyword in query_lower for keyword in ["symptoms", "diagnosis", "condition", "disease", "test results", "medical history", "exam", "patient case", "chest pain", "shortness of breath", "fatigue", "ecg", "troponin", "cardiac"]): # Added keywords from recent query
            return "diagnosis" # Changed from clinical to diagnosis for specificity
        elif any(keyword in query_lower for keyword in ["safety", "interaction", "side effects", "contraindication", "allergy", "risk"]):
            return "safety"
        else:
            return "general_clinical" # Default to a more generic clinical domain

    def generate_response(self, question: str, patient_info: Optional[Dict] = None, domain: Optional[str] = None) -> Dict:
        start_time_total = datetime.now()
        global SAFETY_RULES #

        # Load all models at the start
        if not self.load_model(): #
            return {
                "response": "AI system is not available", #
                "error": "Model failed to load" #
            }

        if domain is None:
            domain = self._determine_domain(question) #

        medical_context = self._get_medical_context(question, domain) #
        prompt_parts = self._construct_prompt(question, domain, patient_info, medical_context) #

        response_text = "" #
        generation_params = { #
            'temperature': 0.7, #
            'top_p': 0.9, #
            'top_k': 40 #
        }
        model_outputs_for_confidence = {} #

        # Select the model based on domain
        current_model = None
        if domain in ["clinical", "diagnosis", "prescription", "safety"] and self.clinical_vertexai_model:
            current_model = self.clinical_vertexai_model
            print(f"Using clinical model for domain: {domain}")
        else:
            current_model = self.vertexai_model
            print(f"Using general model for domain: {domain}")

        if current_model is None:
            return {
                "response": "I cannot provide a response as the appropriate model could not be loaded.",
                "error": "No valid model available for the detected domain."
            }

        try:
            print(f"🔍 Processing query via Vertex AI API: {question}...")
            if isinstance(current_model, GenerativeModel):
                response = current_model.generate_content(
                    contents=prompt_parts,
                    safety_settings=SAFETY_RULES, #
                    generation_config={ #
                        "temperature": 0.5, #
                        "max_output_tokens": 3024, #
                    }
                )
                response_text = response.text
            elif isinstance(current_model, aiplatform.Endpoint):
                flat_prompt = ""
                for part_dict in prompt_parts:
                    if 'parts' in part_dict and isinstance(part_dict['parts'], list):
                        flat_prompt += " ".join(str(p) for p in part_dict['parts']) + "\n"
                prediction_instances = [{"prompt_text": flat_prompt}]
                endpoint_parameters = {
                    'maxOutputTokens': 1024,
                    'temperature': generation_params['temperature'],
                    'topP': generation_params['top_p'],
                    'topK': generation_params['top_k']
                }
                response = current_model.predict(
                    instances=prediction_instances,
                    parameters=endpoint_parameters
                )
                response_text = response.predictions[0]
            elif isinstance(current_model, vertexai.preview.language_models.TextGenerationModel):
                flat_prompt = ""
                for part_dict in prompt_parts:
                    if 'parts' in part_dict and isinstance(part_dict['parts'], list):
                        flat_prompt += " ".join(str(p) for p in part_dict['parts']) + "\n"
                response = current_model.predict(
                    prompt=flat_prompt,
                    max_output_tokens=1024,
                    temperature=generation_params['temperature'],
                    top_p=generation_params['top_p'],
                    top_k=generation_params['top_k']
                )
                response_text = response.text
            else:
                raise TypeError("Unsupported model type loaded in vertexai_model.")

            if not response_text:
                raise ValueError("Vertex AI API generated empty output")

            model_outputs_for_confidence = {
                'generated_text': response_text,
                'vertexai_response': response
            }

        except Exception as e:
            response_text = f"I cannot provide a response for that query due to ERROR: {e}. Please rephrase your question."
            print(f"❌ Error during Vertex AI API call: {e}")
            model_outputs_for_confidence = {'generated_text': response_text}

        initial_confidence_for_safety = 0.5 #
        safety_evaluation = self.safety_system.comprehensive_safety_check( #
            recommendation=response_text, #
            domain=domain, #
            confidence_score=initial_confidence_for_safety, #
            patient_info=patient_info #
        )

        confidence_scores = self.confidence_scorer.calculate_overall_confidence( #
            model_outputs=model_outputs_for_confidence, #
            response=response_text, #
            query=question, #
            domain=domain, #
            safety_assessment=safety_evaluation, #
            patient_info=patient_info, #
            generation_params=generation_params #
        )

        visuals = {} #
        if hasattr(self, 'visualizer') and PLOTLY_AVAILABLE: #
            try: #
                confidence_viz = None #
                safety_viz = None #
                if hasattr(self.visualizer, 'plot_confidence_breakdown'): #
                    confidence_viz = self.visualizer.plot_confidence_breakdown(confidence_scores) #
                if hasattr(self.visualizer, 'generate_safety_report'): #
                    safety_viz = self.visualizer.generate_safety_report(safety_evaluation) #

                if confidence_viz: #
                    visuals['confidence_visualization'] = confidence_viz #
                if safety_viz: #
                    visuals['safety_report_visualization'] = safety_viz #
            except Exception as e: #
                print(f"Error generating visualizations: {e}") #

        end_time_total = datetime.now() #
        processing_time = (end_time_total - start_time_total).total_seconds() #
        print(f"Overall processing time: {processing_time:.2f} seconds") #

        return { #
            "response": response_text, #
            "domain": domain, #
            "confidence": confidence_scores, #
            "safety_assessment": safety_evaluation, #
            "visuals": visuals, #
            "processing_time_seconds": processing_time #
        }
# ==============================================
# 🚀 ENHANCED UNIFIED HEALTHCARE AI SYSTEM (Demonstration)
# ==============================================

class EnhancedUnifiedHealthcareAI(HealthcareAISystem):
    def __init__(self, model_name: str = "gemini-2.0-flash-lite-001", tuned_model_id: str = None):
        super().__init__(model_name, tuned_model_id)
        self.cache = {}
        self.recommendation_engine = self._load_recommendation_engine()
        self.session_analytics = {
            "total_queries": 0,
            "cache_hits": 0,
            "processing_times": [],
            "confidence_scores": [],
            "safety_risks": []
        }

    def _load_recommendation_engine(self) -> Dict:
        """Load a simple recommendation mapping."""
        return {
            "diabetes_diet": ["low-carb meals", "regular exercise", "monitor blood sugar"],
            "hypertension_medication": ["ACE inhibitors", "ARBs", "diuretics"],
            "flu_symptoms": ["rest", "fluids", "over-the-counter pain relievers"],
            "general_wellness": ["balanced diet", "regular physical activity", "adequate sleep"]
        }

    def process_query(self, query: str, patient_info: Optional[Dict] = None) -> Dict:
        """Process a query with caching and enhanced features."""
        self.session_analytics["total_queries"] += 1

        if query in self.cache:
            self.session_analytics["cache_hits"] += 1
            print("Cache hit!")
            cached_response = self.cache[query]
            self.session_analytics["processing_times"].append(cached_response.get("processing_time_seconds", 0))
            self.session_analytics["confidence_scores"].append(cached_response["confidence"]["overall_confidence"])
            self.session_analytics["safety_risks"].append(cached_response["safety_assessment"]["risk_level"])
            return cached_response

        # Use the base class's generate_response
        response_data = self.generate_response(query, patient_info)
        
        # Post-processing for recommendations
        domain = response_data.get("domain", "general_clinical")
        recommendations = self._generate_recommendations(query, domain)
        response_data["related_recommendations"] = recommendations

        self.cache[query] = response_data
        self.session_analytics["processing_times"].append(response_data.get("processing_time_seconds", 0))
        self.session_analytics["confidence_scores"].append(response_data["confidence"]["overall_confidence"])
        self.session_analytics["safety_risks"].append(response_data["safety_assessment"]["risk_level"])

        return response_data

    def _generate_recommendations(self, query: str, domain: str) -> List[str]:
        """Generate context-aware recommendations."""
        recommendations = []
        query_lower = query.lower()

        if "diabetes" in query_lower:
            recommendations.extend(self.recommendation_engine.get("diabetes_diet", []))
        if "hypertension" in query_lower:
            recommendations.extend(self.recommendation_engine.get("hypertension_medication", []))
        if "flu" in query_lower or "cold" in query_lower:
            recommendations.extend(self.recommendation_engine.get("flu_symptoms", []))
        
        # Fallback to general wellness recommendations if no specific ones apply
        if not recommendations:
            recommendations.extend(self.recommendation_engine.get("general_wellness", []))

        return list(set(recommendations)) # Return unique recommendations

    def batch_process(self, queries: List[str], patient_info_list: Optional[List[Dict]] = None) -> List[Dict]:
        """Process multiple queries efficiently."""
        results = []
        for i, query in enumerate(queries):
            patient_info = patient_info_list[i] if patient_info_list and i < len(patient_info_list) else None
            results.append(self.process_query(query, patient_info))
        return results

    def get_session_analytics(self) -> Dict:
        """Get performance metrics for the current session."""
        avg_processing_time = np.mean(self.session_analytics["processing_times"]) if self.session_analytics["processing_times"] else 0
        avg_confidence = np.mean(self.session_analytics["confidence_scores"]) if self.session_analytics["confidence_scores"] else 0
        max_safety_risk = max(self.session_analytics["safety_risks"]) if self.session_analytics["safety_risks"] else 0

        return {
            "total_queries": self.session_analytics["total_queries"],
            "cache_hits": self.session_analytics["cache_hits"],
            "cache_hit_ratio": self.session_analytics["cache_hits"] / self.session_analytics["total_queries"] if self.session_analytics["total_queries"] > 0 else 0,
            "average_processing_time_seconds": avg_processing_time,
            "average_confidence_score": avg_confidence,
            "highest_safety_risk_level_encountered": max_safety_risk
        }

def demo_enhanced_system():
    enhanced_ai_system = EnhancedUnifiedHealthcareAI()
    model_loaded = enhanced_ai_system.load_model()
    print("Enhanced Model loaded successfully:", model_loaded)

    if not model_loaded:
        print("⚠️ Enhanced system running in fallback mode due to model loading failure.")
        return

    # Demo 1: Basic query
    print("\n--- Demo 1: Basic Query ---")
    result1 = enhanced_ai_system.process_query(
        "What are the symptoms of a common cold?",
        {"age": 30, "conditions": []}
    )
    print(f"Response: {result1['response']}")
    print(f"Confidence: {result1['confidence']['overall_confidence']:.1%}")
    print(f"Safety: {result1['safety_assessment']['message']}")
    print(f"Recommendations: {result1['related_recommendations']}")

    # Demo 2: Cached query
    print("\n--- Demo 2: Cached Query ---")
    result2 = enhanced_ai_system.process_query(
        "What are the symptoms of a common cold?",
        {"age": 30, "conditions": []}
    )
    print(f"Response: {result2['response']}")
    print(f"Confidence: {result2['confidence']['overall_confidence']:.1%}")
    print(f"Safety: {result2['safety_assessment']['message']}")
    print(f"Recommendations: {result2['related_recommendations']}")

    # Demo 3: Nutritional query with patient info
    print("\n--- Demo 3: Nutritional Query with Patient Info ---")
    result3 = enhanced_ai_system.process_query(
        "What should a diabetic patient eat for breakfast?",
        {"age": 55, "conditions": ["diabetes"], "pregnant": False}
    )
    print(f"Response: {result3['response']}")
    print(f"Confidence: {result3['confidence']['overall_confidence']:.1%}")
    print(f"Safety: {result3['safety_assessment']['message']}")
    print(f"Recommendations: {result3['related_recommendations']}")

    # Demo 4: Batch processing
    print("\n--- Demo 4: Batch Processing ---")
    batch_queries = [
        "How to manage high blood pressure?",
        "What are good sources of protein?",
        "Symptoms of heart attack?"
    ]
    batch_patient_info = [
        {"age": 60, "conditions": ["hypertension"]},
        {"age": 25, "conditions": []},
        {"age": 70, "conditions": ["heart disease history"]}
    ]
    batch_results = enhanced_ai_system.batch_process(batch_queries, batch_patient_info)
    for i, res in enumerate(batch_results):
        print(f"\nBatch Query {i+1}: {batch_queries[i]}")
        print(f"  Response: {res['response']}")
        print(f"  Confidence: {res['confidence']['overall_confidence']:.1%}")
        print(f"  Safety: {res['safety_assessment']['message']}")

    # Demo 5: Session Analytics
    print("\n--- Demo 5: Session Analytics ---")
    analytics = enhanced_ai_system.get_session_analytics()
    print(f"Total Queries: {analytics['total_queries']}")
    print(f"Cache Hits: {analytics['cache_hits']} ({analytics['cache_hit_ratio']:.1%})")
    print(f"Average Processing Time: {analytics['average_processing_time_seconds']:.2f} seconds")
    print(f"Average Confidence Score: {analytics['average_confidence_score']:.1%}")
    print(f"Highest Safety Risk Level: {analytics['highest_safety_risk_level_encountered']}")


if __name__ == "__main__":
    print("==================================================")
    print("🚀 DEMOING COMPLETE UNIFIED HEALTHCARE AI SYSTEM")
    print("==================================================")

    print("\n1️⃣ BASIC HEALTHCARE AI SYSTEM")
    print("-" * 30)
    ai_system = HealthcareAISystem()
    model_loaded = ai_system.load_model()
    print("Model loaded successfully:", model_loaded)
    
    if model_loaded:
        print("✅ Basic Healthcare AI System ready!")
    else:
        print("⚠️ Running in fallback mode")
    
    # Quick demo
    result = ai_system.generate_response(
        "What are the symptoms of diabetes?",
        {"age": 40, "conditions": []}
    )
    
    print(f"Sample response confidence: {result['confidence']['overall_confidence']:.1%}")
    print(f"Domain identified: {result['domain']}")
    
    # Demo enhanced system
    print("\n2️⃣ ENHANCED UNIFIED HEALTHCARE AI SYSTEM")
    print("-" * 30)
    
    demo_enhanced_system()
    
    print("\n" + "="*60)
    print("💡 USAGE INSTRUCTIONS:")
    print("="*60)
    print("Basic System:")
    print("  • HealthcareAISystem() - Initialize basic system")
    print("  • generate_response() - Get AI response with confidence")
    print("  • generate_report() - Get comprehensive report")
    print("  • fine_tune_model() - Fine-tune on your data")
    
    print("\nEnhanced System:")
    print("  • EnhancedUnifiedHealthcareAI() - Initialize enhanced system")
    print("  • process_query() - Process with caching and recommendations")
    print("  • batch_process() - Handle multiple queries efficiently")
    print("  • get_session_analytics() - Get performance metrics")

    print("\n" + "="*60)
    print("Generated by User Request:")
    print("="*60)

def generate():
  client = genai.Client(
      vertexai=True,
      project="729813973979",
      location="us-central1",
  )

  model = "projects/729813973979/locations/us-central1/endpoints/1748994245814910976"
  contents = [
    types.Content(
      role="user",
      parts=[
          genai.types.Part.from_text("What are the benefits of regular exercise?") # Added a sample prompt
      ]
    )
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

if __name__ == "__main__":
    # Call the newly added generate function for demonstration
    print("\n--- Calling user-provided generate() function ---")
    generate()