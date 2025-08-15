import pandas as pd
import numpy as np
import random
from datetime import datetime, date, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy import stats
import hashlib
from functools import lru_cache
import json
from collections import defaultdict

# Ignore FutureWarnings from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class Person:
    """A data class to represent a person with core demographic information."""
    person_id: str
    age: int
    sex: str
    nationality: str = 'Uruguayan'
    healthcare_provider: str = 'public'
    geographic_cluster_id: str = 'G1'
    overall_risk_category: Optional[str] = None
    high_healthcare_utilizer: Optional[bool] = None
    socioeconomic_status_score: Optional[float] = None
    area_type: Optional[str] = None
    lives_alone: Optional[bool] = None
    marital_status: Optional[str] = None
    employment_status: Optional[str] = None
    # Use field for mutable default values
    additional_data: Dict[str, Any] = field(default_factory=dict)
    # Track previous features for consistency
    historical_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealismConfig:
    """Configuration for controlling realism aspects of generated data."""
    correlation_strength: float = 0.7  # How strongly features correlate
    noise_level: float = 0.1          # Random variation
    missing_data_rate: float = 0.05   # Proportion of missing values
    outlier_rate: float = 0.02        # Proportion of outliers
    temporal_consistency: bool = True  # Maintain consistency over time
    seasonal_effects: bool = True     # Include seasonal patterns
    covid_impact: bool = True         # Include pandemic effects
    data_quality_variation: bool = True # Vary data quality by demographics

class Config:
    """Centralized configuration for all feature generation parameters."""
    EDUCATION_LEVELS = {
        'no_formal_education': 0.05, 'primary_incomplete': 0.15, 'primary_complete': 0.20,
        'secondary_incomplete': 0.25, 'secondary_complete': 0.20,
        'tertiary_incomplete': 0.10, 'tertiary_complete': 0.05
    }
    
    EMPLOYMENT_STATUS = {
        'employed_full_time': 0.45, 'employed_part_time': 0.15, 'unemployed_seeking': 0.12,
        'unemployed_not_seeking': 0.08, 'student': 0.10, 'retired': 0.08, 'disabled': 0.02
    }
    
    # New: Employment multipliers based on education
    EMPLOYMENT_MULTIPLIERS = {
        'tertiary_complete': {'employed_full_time': 1.5, 'unemployed_seeking': 0.3},
        'tertiary_incomplete': {'employed_full_time': 1.3, 'unemployed_seeking': 0.4},
        'secondary_complete': {'employed_full_time': 1.0, 'unemployed_seeking': 1.0},
        'secondary_incomplete': {'employed_full_time': 0.8, 'unemployed_seeking': 1.2},
        'primary_complete': {'employed_full_time': 0.6, 'unemployed_seeking': 1.5},
        'primary_incomplete': {'employed_full_time': 0.4, 'unemployed_seeking': 2.0},
        'no_formal_education': {'employed_full_time': 0.3, 'unemployed_seeking': 2.5}
    }
    
    MARITAL_STATUS = {
        'single': 0.35, 'married': 0.40, 'divorced': 0.12,
        'separated': 0.05, 'widowed': 0.08
    }
    
    AREA_CLASSIFICATION = {
        'urban': 0.70, 'suburban': 0.20, 'rural': 0.10
    }
    
    MH_CONDITIONS = {
        'major_depression': 0.08, 'anxiety_disorders': 0.12, 'bipolar_disorder': 0.03,
        'personality_disorders': 0.05, 'substance_use_disorder': 0.10,
        'psychotic_disorders': 0.01, 'eating_disorders': 0.02, 'ptsd': 0.04
    }
    
    # New: Chronic conditions that persist over time
    CHRONIC_CONDITIONS = {
        'diabetes', 'hypertension', 'heart_disease', 'chronic_kidney_disease',
        'neurological_disorder', 'bipolar_disorder', 'personality_disorders'
    }
    
    PHYSICAL_CONDITIONS = {
        'diabetes': 0.09, 'hypertension': 0.25, 'heart_disease': 0.06,
        'chronic_pain': 0.15, 'cancer_history': 0.04, 'neurological_disorder': 0.03,
        'chronic_kidney_disease': 0.02, 'respiratory_disease': 0.08
    }
    
    MEDICATION_CLASSES = {
        'antidepressants': 0.15, 'anxiolytics': 0.12, 'antipsychotics': 0.03,
        'mood_stabilizers': 0.02, 'stimulants': 0.04, 'anticonvulsants': 0.05,
        'sedatives': 0.08
    }
    
    RISK_FACTORS = {
        'hopelessness_scale_score': {'dist': 'normal', 'mean': 12, 'std': 6, 'range': (0, 20)},
        'depression_severity_phq9': {'dist': 'normal', 'mean': 15, 'std': 5, 'range': (0, 27)},
        'anxiety_severity_gad7': {'dist': 'normal', 'mean': 12, 'std': 4, 'range': (0, 21)},
        'suicidal_ideation_intensity': {'dist': 'normal', 'mean': 6, 'std': 3, 'range': (0, 10)},
        'social_connectedness_score': {'dist': 'normal', 'mean': 4, 'std': 2, 'range': (0, 10)},
        'life_stress_score': {'dist': 'normal', 'mean': 7, 'std': 2, 'range': (0, 10)}
    }
    
    PROTECTIVE_FACTORS = {
        'reasons_for_living_score': {'dist': 'normal', 'mean': 35, 'std': 8, 'range': (0, 48)},
        'coping_skills_score': {'dist': 'normal', 'mean': 25, 'std': 6, 'range': (0, 40)},
        'treatment_alliance_score': {'dist': 'normal', 'mean': 7, 'std': 2, 'range': (0, 10)}
    }
    
    BINARY_RISK_INDICATORS = {
        'recent_loss_bereavement': 0.15, 'relationship_breakup_recent': 0.20, 'job_loss_recent': 0.12,
        'legal_problems': 0.08, 'financial_crisis': 0.25, 'academic_failure': 0.10,
        'social_isolation': 0.30, 'bullying_victim': 0.08, 'domestic_violence_exposure': 0.12,
        'childhood_trauma_history': 0.35
    }
    
    ACE_CATEGORIES = {
        'physical_abuse_childhood': 0.28, 'emotional_abuse_childhood': 0.35, 'sexual_abuse_childhood': 0.20,
        'physical_neglect_childhood': 0.16, 'emotional_neglect_childhood': 0.18,
        'household_dysfunction_childhood': 0.25, 'parental_substance_abuse': 0.30,
        'parental_mental_illness': 0.20, 'domestic_violence_witnessed': 0.12
    }
    
    ADULT_TRAUMA_TYPES = {
        'combat_exposure': 0.05, 'serious_accident': 0.15, 'natural_disaster': 0.08,
        'violent_crime_victim': 0.12, 'sexual_assault_adult': 0.10
    }
    
    QUALITY_INDICATORS = {
        'care_coordination_score': {'dist': 'normal', 'mean': 7.5, 'std': 2.0, 'range': (0, 10)},
        'treatment_accessibility_score': {'dist': 'normal', 'mean': 6.8, 'std': 2.5, 'range': (0, 10)},
        'provider_communication_score': {'dist': 'normal', 'mean': 8.0, 'std': 1.5, 'range': (0, 10)},
        'continuity_of_care_score': {'dist': 'normal', 'mean': 7.2, 'std': 2.2, 'range': (0, 10)}
    }
    
    PERFORMANCE_METRICS = {
        'wait_time_initial_appointment_days': {'dist': 'poisson', 'lambda': 14},
        'treatment_delay_days': {'dist': 'poisson', 'lambda': 7},
        'missed_appointments_count': {'dist': 'poisson', 'lambda': 2},
        'provider_changes_count': {'dist': 'poisson', 'lambda': 1.5}
    }
    
    CARE_SETTINGS = {
        'primary_care': 0.85, 'community_mental_health': 0.60, 'specialist_psychiatric': 0.35,
        'emergency_services': 0.25, 'inpatient_psychiatric': 0.15,
        'intensive_outpatient': 0.20, 'peer_support_services': 0.30
    }
    
    DEVICE_ADOPTION = {
        'smartphone': 0.85, 'fitness_tracker': 0.35, 'smartwatch': 0.25,
        'sleep_tracker': 0.15, 'mood_app': 0.20
    }
    
    # New: Biomarker categories for better organization
    BIOMARKER_CATEGORIES = {
        'sleep_metrics': ['avg_sleep_duration_hours', 'sleep_efficiency_percent', 'sleep_latency_minutes'],
        'activity_metrics': ['daily_steps_avg', 'active_minutes_daily', 'heart_rate_variability'],
        'digital_behavior': ['screen_time_hours_daily', 'app_switches_daily', 'typing_speed_wpm'],
        'communication': ['call_frequency_daily', 'text_message_frequency_daily', 'social_contact_diversity'],
        'environmental': ['ambient_light_avg_lux', 'ambient_noise_avg_db', 'air_quality_index']
    }
    
    ML_CONFIGS = {
        'encoding_schemes': ['ordinal', 'one_hot', 'target', 'frequency'],
        'scaling_methods': ['standard', 'minmax', 'robust', 'quantile'],
        'imputation_strategies': ['mean', 'median', 'mode', 'knn', 'iterative'],
        'cv_strategies': ['stratified_kfold', 'time_series_split', 'group_kfold'],
        'n_folds_options': [3, 5, 10],
        'model_families': [
            'logistic_regression', 'random_forest', 'gradient_boosting',
            'neural_network', 'svm', 'naive_bayes', 'ensemble'
        ]
    }
    
    # New: Seasonal adjustment factors
    SEASONAL_ADJUSTMENTS = {
        'depression_severity_phq9': {11: 1.2, 12: 1.3, 1: 1.4, 2: 1.2},  # Winter months
        'social_isolation': {11: 1.1, 12: 1.2, 1: 1.3, 2: 1.1},
        'substance_use_disorder': {11: 1.1, 12: 1.3, 1: 1.2}  # Holiday effects
    }
    
    # New: COVID impact periods
    COVID_PERIODS = {
        'initial_lockdown': (date(2020, 3, 15), date(2020, 6, 1)),
        'second_wave': (date(2020, 10, 1), date(2021, 2, 28)),
        'recovery_phase': (date(2021, 6, 1), date(2022, 12, 31))
    }

def auto_feature_names(cls: Any) -> Any:
    """Enhanced decorator to automatically generate get_feature_names from _distributions."""
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, '_distributions'):
            self._feature_names = []
            for group_key, group in self._distributions.items():
                if isinstance(group, dict):
                    if all('dist' in v for v in group.values() if isinstance(v, dict)):
                        self._feature_names.extend(list(group.keys()))
                    else:
                        prefix = 'has_' if 'conditions' in group_key else ''
                        self._feature_names.extend([f'{prefix}{k}' for k in group.keys()])
                elif isinstance(group, list):
                    for item in group:
                        self._feature_names.extend([
                            f'uses_{item}_app', f'{item}_sessions_per_week', 
                            f'{item}_avg_session_minutes', f'{item}_adherence_rate'
                        ])
            
            # Enhanced special cases with more comprehensive coverage
            special_cases = {
                'SociodemographicGenerator': [
                    'socioeconomic_status_score', 'social_support_score', 'lives_alone', 
                    'primary_earner', 'household_size', 'children_in_household'
                ],
                'ClinicalComorbidityGenerator': [
                    'polypharmacy', 'medication_count', 'gaf_score', 'clinical_severity', 
                    'functional_impairment', 'comorbidity_count_mental', 'complex_case', 
                    'ed_visits_past_year', 'hospitalizations_past_year', 'outpatient_visits_past_year',
                    'high_healthcare_utilizer', 'total_healthcare_encounters', 'charlson_comorbidity_index',
                    'mental_health_burden_score', 'clinical_complexity', 'total_comorbidity_burden',
                    'chronic_condition_stability', 'medication_adherence_score'
                ],
                'RiskFactorGenerator': [
                    'ace_score', 'high_ace_score', 'adult_trauma_count', 'polytraumatized', 
                    'weighted_risk_index', 'protective_factor_index', 'net_risk_score', 
                    'acute_stressor_count', 'acute_risk_elevation', 'overall_risk_category', 
                    'risk_protective_ratio', 'seasonal_depression_risk', 'covid_impact_factor'
                ],
                'HealthcareSystemGenerator': [
                    'travel_distance_to_care_km', 'transportation_barriers', 'same_day_access_available',
                    'after_hours_access_available', 'telehealth_available', 'language_barrier_present',
                    'cultural_competency_rating', 'ehr_integrated', 'care_plan_documented', 
                    'crisis_plan_available', 'multidisciplinary_team', 'case_manager_assigned',
                    'peer_support_available', 'provider_communication_quality', 'discharge_planning_quality',
                    'primary_insurance', 'mental_health_coverage_adequate', 'medication_coverage_adequate',
                    'therapy_sessions_covered', 'financial_barriers_to_care', 'medication_cost_burden',
                    'delayed_care_due_to_cost', 'skipped_medications_due_to_cost', 'data_quality_score'
                ],
                'MachineLearningGenerator': [
                    'temporal_split', 'random_split', 'stratified_split', 'geographic_split', 'cv_fold',
                    'is_train_set', 'is_validation_set', 'is_test_set', 'categorical_encoding_strategy',
                    'numerical_scaling_method', 'missing_value_strategy', 'high_cardinality_features',
                    'sparse_features_count', 'engineered_features_count', 'data_quality_flag',
                    'outlier_score', 'missing_data_percentage', 'feature_correlation_max',
                    'target_correlation', 'variance_inflation_factor', 'primary_model_family',
                    'ensemble_method', 'cv_strategy', 'cv_folds', 'hyperparameter_tuning_method',
                    'optimization_metric', 'training_time_minutes', 'convergence_achieved',
                    'early_stopping_triggered', 'model_consensus_score', 'prediction_stability',
                    'ensemble_weight', 'calibration_score', 'over_confident', 'under_confident',
                    'local_explanation_confidence', 'counterfactual_distance', 'model_complexity_score',
                    'linear_separability', 'feature_interaction_strength', 'clinical_rule_triggered',
                    'recommendation_type', 'epistemic_uncertainty', 'aleatoric_uncertainty', 
                    'total_uncertainty'
                ],
                'BiometricsDigitalGenerator': [
                    'digital_device_count', 'high_digital_engagement', 'digital_biomarker_completeness',
                    'mental_health_app_count', 'digital_therapy_engaged'
                ]
            }
            
            if cls.__name__ in special_cases:
                self._feature_names.extend(special_cases[cls.__name__])
            
            # Add prediction horizons dynamically
            if cls.__name__ == 'MachineLearningGenerator':
                horizons = ['7_day', '30_day', '90_day', '6_month', '1_year']
                for horizon in horizons:
                    self._feature_names.extend([
                        f'prediction_score_{horizon}', 
                        f'prediction_lower_ci_{horizon}', 
                        f'prediction_upper_ci_{horizon}'
                    ])
            
            # Add all biomarker categories dynamically
            if cls.__name__ == 'BiometricsDigitalGenerator':
                for category, biomarkers in Config.BIOMARKER_CATEGORIES.items():
                    self._feature_names.extend(biomarkers)

    cls.__init__ = new_init

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    cls.get_feature_names = get_feature_names
    return cls

class BaseFeatureGenerator(ABC):
    """Enhanced abstract base class for all feature generators."""

    def __init__(self, realism_config: Optional[RealismConfig] = None):
        self._distributions: Dict[str, Any] = {}
        self._feature_names: List[str] = []
        self.realism_config = realism_config or RealismConfig()
        self._cache = {}

    def _generate_marital_status(self, age: int, sex: str) -> str:
        """Generate marital status with age and sex patterns."""
        if age < 16: return 'single'
        
        if age < 25: probs = [0.80, 0.15, 0.03, 0.02, 0.00]
        elif age < 35: probs = [0.45, 0.45, 0.05, 0.04, 0.01]
        elif age < 50: probs = [0.25, 0.55, 0.12, 0.06, 0.02]
        elif age < 65: probs = [0.20, 0.50, 0.15, 0.08, 0.07]
        else: probs = [0.15, 0.35, 0.10, 0.05, 0.35]
        
        if sex == 'F' and age > 65: probs[4] *= 1.5
        
        return np.random.choice(list(Config.MARITAL_STATUS.keys()), p=np.array(probs) / sum(probs))

    def _generate_household_size(self, age: int, marital: str) -> int:
        if marital == 'single' and age > 25: return np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        if marital in ['married', 'separated']: return np.random.choice([2, 3, 4, 5, 6], p=[0.3, 0.25, 0.25, 0.15, 0.05])
        if marital == 'divorced': return np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        if marital == 'widowed': return np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        return np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])
        
    def _generate_children_count(self, age: int, marital: str) -> int:
        if age < 20 or marital == 'single': return np.random.choice([0, 1], p=[0.8, 0.2])
        if marital in ['married', 'separated', 'divorced']: return np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        return np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])

    def _calculate_ses_score(self, education: str, employment: str, area_type: str) -> float:
        education_scores = {
            'no_formal_education': 0.0, 'primary_incomplete': 0.2, 'primary_complete': 0.3,
            'secondary_incomplete': 0.5, 'secondary_complete': 0.7,
            'tertiary_incomplete': 0.8, 'tertiary_complete': 1.0
        }
        employment_scores = {
            'employed_full_time': 1.0, 'employed_part_time': 0.7, 'unemployed_seeking': 0.2,
            'unemployed_not_seeking': 0.1, 'student': 0.5, 'retired': 0.6, 'disabled': 0.3
        }
        area_scores = {'urban': 0.8, 'suburban': 0.6, 'rural': 0.4}
        
        education_weight, employment_weight, area_weight = 0.5, 0.3, 0.2
        return (education_scores[education] * education_weight +
                employment_scores[employment] * employment_weight +
                area_scores[area_type] * area_weight)

    def _generate_social_support_score(self, marital: str, household_size: int) -> float:
        base_score = 0.5
        marital_adjustments = {
            'married': 0.3, 'single': -0.1, 'divorced': -0.2,
            'separated': -0.2, 'widowed': -0.1
        }
        household_adjustment = min(0.2, (household_size - 1) * 0.1)
        support_score = base_score + marital_adjustments.get(marital, 0) + household_adjustment
        support_score += np.random.normal(0, 0.1)
        return max(0.0, min(1.0, support_score))

    def _is_primary_earner(self, employment: str, marital: str, sex: str) -> bool:
        if employment in ['unemployed_seeking', 'unemployed_not_seeking', 'disabled']: return False
        if employment == 'employed_full_time': return True
        if marital == 'single': return employment in ['employed_full_time', 'employed_part_time']
        return np.random.random() < (0.6 if sex == 'M' else 0.4)


@auto_feature_names
class ClinicalComorbidityGenerator(BaseFeatureGenerator):
    """Enhanced clinical comorbidity features with temporal consistency."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'mental_health_conditions': Config.MH_CONDITIONS,
            'physical_conditions': Config.PHYSICAL_CONDITIONS,
            'medication_classes': Config.MEDICATION_CLASSES,
        }

    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        age, sex = person.age, person.sex

        # Helper functions for adjustments
        def adjust_mh_rate(condition: str, base_rate: float) -> float:
            adjusted_rate = base_rate
            if condition == 'eating_disorders' and 15 <= age <= 25: adjusted_rate *= 3.0
            elif condition == 'anxiety_disorders' and 25 <= age <= 45: adjusted_rate *= 1.5
            elif condition == 'major_depression' and age > 65: adjusted_rate *= 1.3
            if sex == 'F':
                if condition in ['major_depression', 'anxiety_disorders', 'eating_disorders']: adjusted_rate *= 1.8
                elif condition == 'substance_use_disorder': adjusted_rate *= 0.6
            else:
                if condition == 'substance_use_disorder': adjusted_rate *= 1.5
                elif condition in ['major_depression', 'anxiety_disorders']: adjusted_rate *= 0.7
            return min(0.8, adjusted_rate * 3.0)

        def adjust_physical_rate(condition: str, base_rate: float) -> float:
            age_multiplier = 0.3 if age < 30 else 0.8 if age < 50 else 1.5 if age < 65 else 2.5
            if condition == 'diabetes' and age < 40: age_multiplier *= 0.5
            elif condition == 'hypertension' and age < 35: age_multiplier *= 0.3
            elif condition == 'chronic_pain': age_multiplier = 1.0
            
            sex_multiplier = 1.0
            if sex == 'F' and condition == 'heart_disease': sex_multiplier = 0.7
            elif sex == 'M' and condition == 'chronic_pain': sex_multiplier = 0.8
            
            return min(0.7, base_rate * age_multiplier * sex_multiplier)

        def adjust_medication_rate(med_class: str, base_rate: float) -> float:
            if med_class == 'antidepressants' and mh_conditions.get('has_major_depression'): return 0.8
            if med_class == 'anxiolytics' and mh_conditions.get('has_anxiety_disorders'): return 0.6
            if med_class == 'antipsychotics' and mh_conditions.get('has_psychotic_disorders'): return 0.9
            if med_class == 'mood_stabilizers' and mh_conditions.get('has_bipolar_disorder'): return 0.7
            return base_rate

        mh_conditions = self._generate_weighted_features(Config.MH_CONDITIONS, adjust_mh_rate)
        physical_conditions = self._generate_weighted_features(Config.PHYSICAL_CONDITIONS, adjust_physical_rate)
        medications = self._generate_weighted_features(Config.MEDICATION_CLASSES, adjust_medication_rate)
        
        # Combine all features and calculate composites
        features = {
            **{f'has_{k}': v for k, v in mh_conditions.items()},
            **{f'has_{k}': v for k, v in physical_conditions.items()},
            **{f'prescribed_{k}': v for k, v in medications.items()}
        }
        
        features.update(self._generate_derived_features(features, age))
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features

    def _generate_derived_features(self, features: Dict, age: int) -> Dict:
        mh_count = sum(v for k, v in features.items() if 'has_' in k and any(mh in k for mh in Config.MH_CONDITIONS))
        phys_count = sum(v for k, v in features.items() if 'has_' in k and any(ph in k for ph in Config.PHYSICAL_CONDITIONS))
        
        # Clinical Severity
        gaf_score = int(np.random.normal(55 if mh_count > 0 else 75, 15 if mh_count > 0 else 10))
        gaf_score = max(1, min(100, gaf_score))
        severity = 'severe' if mh_count >= 3 else 'moderate' if mh_count >= 2 else 'mild' if mh_count >= 1 else 'minimal'
        
        # Healthcare Utilization
        total_conditions = mh_count + phys_count
        ed_visits = np.random.poisson(0.5 + total_conditions * 0.3)
        hosp = np.random.poisson(0.1 + total_conditions * 0.15)
        outpatient = np.random.poisson(3 + total_conditions * 2)
        
        # Enhanced medication adherence scoring
        medication_count = sum(v for k, v in features.items() if 'prescribed_' in k)
        medication_adherence_score = self._calculate_medication_adherence(medication_count, mh_count)
        
        # Chronic condition stability (new feature)
        chronic_condition_stability = self._assess_chronic_stability(features)
        
        # Composite Scores
        charlson = phys_count
        mh_burden = mh_count
        total_burden = charlson + mh_burden
        complexity = 'high' if total_burden >= 5 else 'moderate' if total_burden >= 3 else 'low' if total_burden >= 1 else 'minimal'

        return {
            'polypharmacy': medication_count >= 3,
            'medication_count': medication_count,
            'gaf_score': gaf_score,
            'clinical_severity': severity,
            'functional_impairment': gaf_score < 60,
            'comorbidity_count_mental': mh_count,
            'complex_case': mh_count >= 2 and gaf_score < 60,
            'ed_visits_past_year': ed_visits,
            'hospitalizations_past_year': hosp,
            'outpatient_visits_past_year': outpatient,
            'high_healthcare_utilizer': ed_visits >= 4 or hosp >= 2 or outpatient >= 15,
            'total_healthcare_encounters': ed_visits + hosp + outpatient,
            'charlson_comorbidity_index': charlson,
            'mental_health_burden_score': mh_burden,
            'clinical_complexity': complexity,
            'total_comorbidity_burden': total_burden,
            'medication_adherence_score': round(medication_adherence_score, 3),
            'chronic_condition_stability': chronic_condition_stability
        }

    def _calculate_medication_adherence(self, medication_count: int, mh_count: int) -> float:
        """Calculate medication adherence score based on complexity."""
        base_adherence = 0.8
        if medication_count > 5:
            base_adherence -= 0.2
        if mh_count > 2:
            base_adherence -= 0.1
        
        # Add random variation
        adherence = base_adherence + np.random.normal(0, 0.15)
        return max(0.0, min(1.0, adherence))

    def _assess_chronic_stability(self, features: Dict) -> str:
        """Assess stability of chronic conditions."""
        chronic_count = sum(1 for condition in Config.CHRONIC_CONDITIONS 
                          if features.get(f'has_{condition}', False))
        
        if chronic_count == 0:
            return 'stable'
        elif chronic_count <= 2:
            return np.random.choice(['stable', 'fluctuating'], p=[0.7, 0.3])
        else:
            return np.random.choice(['stable', 'fluctuating', 'deteriorating'], p=[0.4, 0.4, 0.2])


@auto_feature_names
class RiskFactorGenerator(BaseFeatureGenerator):
    """Enhanced evidence-based risk and protective factor features."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'risk_factors': Config.RISK_FACTORS,
            'protective_factors': Config.PROTECTIVE_FACTORS,
            'binary_risk_indicators': Config.BINARY_RISK_INDICATORS,
            'ace_categories': Config.ACE_CATEGORIES,
            'adult_trauma_types': Config.ADULT_TRAUMA_TYPES,
        }

    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        age, sex = person.age, person.sex
        
        def adjust_risk_score(factor: str, mean: float, std: float) -> Tuple[float, float]:
            if factor == 'hopelessness_scale_score' and age < 25: mean += 2
            elif factor == 'depression_severity_phq9' and age > 65: mean += 1.5
            elif factor == 'anxiety_severity_gad7' and 25 <= age <= 45: mean += 1
            if sex == 'F':
                if factor in ['depression_severity_phq9', 'anxiety_severity_gad7']: mean += 1.5
            else:
                if factor == 'hopelessness_scale_score': mean += 1
            return mean, std

        def adjust_protective_score(factor: str, mean: float, std: float) -> Tuple[float, float]:
            if factor == 'reasons_for_living_score':
                if age < 20: mean -= 3
                elif age > 60: mean += 2
            elif factor == 'coping_skills_score' and age > 40: mean += 2
            if sex == 'F' and factor == 'social_connectedness_score': mean += 1
            return mean, std
        
        def adjust_event_rate(event: str, base_rate: float) -> float:
            adjusted_rate = base_rate
            if event == 'relationship_breakup_recent' and 16 <= age <= 35: adjusted_rate *= 2.0
            elif event == 'job_loss_recent' and 25 <= age <= 55: adjusted_rate *= 1.5
            elif event == 'academic_failure' and 15 <= age <= 25: adjusted_rate *= 3.0
            elif event == 'financial_crisis' and 30 <= age <= 60: adjusted_rate *= 1.3
            elif event == 'social_isolation' and age > 65: adjusted_rate *= 1.8
            if sex == 'F':
                if event in ['domestic_violence_exposure', 'bullying_victim']: adjusted_rate *= 1.4
            else:
                if event in ['job_loss_recent', 'legal_problems']: adjusted_rate *= 1.2
            return min(0.8, adjusted_rate * 2.5)

        def adjust_trauma_rate(trauma_type: str, base_rate: float) -> float:
            adjusted_rate = base_rate
            if sex == 'F' and trauma_type == 'sexual_abuse_childhood': adjusted_rate *= 2.5
            elif sex == 'M' and trauma_type == 'physical_abuse_childhood': adjusted_rate *= 1.3
            if trauma_type == 'combat_exposure' and sex == 'M': adjusted_rate *= 3.0
            elif trauma_type == 'sexual_assault_adult' and sex == 'F': adjusted_rate *= 4.0
            return min(0.7, adjusted_rate * 2.0)

        risk_scores = self._generate_normal_features(Config.RISK_FACTORS, adjust_risk_score)
        protective_scores = self._generate_normal_features(Config.PROTECTIVE_FACTORS, adjust_protective_score)
        life_events = self._generate_weighted_features(Config.BINARY_RISK_INDICATORS, adjust_event_rate)
        
        # Trauma History
        ace_trauma = self._generate_weighted_features(Config.ACE_CATEGORIES, adjust_trauma_rate)
        adult_trauma = self._generate_weighted_features(Config.ADULT_TRAUMA_TYPES, adjust_trauma_rate)
        ace_score = sum(ace_trauma.values())
        adult_trauma_count = sum(adult_trauma.values())

        # Combine all features
        features = {
            **risk_scores,
            **protective_scores,
            **life_events,
            **ace_trauma,
            **adult_trauma
        }
        features.update({
            'ace_score': ace_score,
            'high_ace_score': ace_score >= 4,
            'adult_trauma_count': adult_trauma_count,
            'polytraumatized': (ace_score >= 2 and adult_trauma_count >= 1)
        })
        
        # Calculate composite risk indices
        composite_indices = self._calculate_risk_indices(risk_scores, protective_scores, life_events)
        person.overall_risk_category = composite_indices['overall_risk_category']
        features.update(composite_indices)
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features

    def _calculate_risk_indices(self, risk_scores: Dict, protective_scores: Dict, life_events: Dict) -> Dict:
        weighted_risk_index = sum([
            risk_scores['hopelessness_scale_score'] * 2.5,
            risk_scores['depression_severity_phq9'] * 1.8,
            risk_scores['suicidal_ideation_intensity'] * 5.0,
            (10 - risk_scores['social_connectedness_score']) * 2.0,
            risk_scores['life_stress_score'] * 2.0
        ])
        weighted_risk_index = min(100, max(0, weighted_risk_index))

        protective_index = sum([
            protective_scores['reasons_for_living_score'] * 1.5,
            protective_scores['coping_skills_score'] * 1.8,
            protective_scores['treatment_alliance_score'] * 4.0
        ])
        protective_index = min(100, max(0, protective_index))
        
        net_risk_score = max(0, weighted_risk_index - (protective_index * 0.7))
        acute_stressors = sum(life_events.values())
        
        risk_category = 'high' if net_risk_score >= 70 or acute_stressors >= 3 else \
                        'moderate' if net_risk_score >= 45 else \
                        'low' if net_risk_score >= 20 else 'minimal'
        
        return {
            'weighted_risk_index': round(weighted_risk_index, 1),
            'protective_factor_index': round(protective_index, 1),
            'net_risk_score': round(net_risk_score, 1),
            'acute_stressor_count': acute_stressors,
            'acute_risk_elevation': acute_stressors >= 3,
            'overall_risk_category': risk_category,
            'risk_protective_ratio': round(weighted_risk_index / max(1, protective_index), 2)
        }


@auto_feature_names
class HealthcareSystemGenerator(BaseFeatureGenerator):
    """Enhanced healthcare system performance and quality indicators."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'quality_indicators': Config.QUALITY_INDICATORS,
            'performance_metrics': Config.PERFORMANCE_METRICS,
            'care_settings': Config.CARE_SETTINGS,
        }

    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        # Quality and performance metrics
        quality_scores = self._generate_quality_indicators(person)
        performance_metrics = self._generate_performance_metrics(person)

        # Access and coverage
        care_access = self._generate_care_access_features(person, attempt_date)
        coverage_features = self._generate_coverage_features(person)
        
        # New: Data quality assessment
        data_quality_features = self._generate_data_quality_features(person)

        features = {
            **quality_scores,
            **performance_metrics,
            **care_access,
            **coverage_features,
            **data_quality_features
        }
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features

    def _generate_quality_indicators(self, person: Person) -> Dict:
        def adjust_quality_score(indicator: str, mean: float, std: float) -> Tuple[float, float]:
            if person.healthcare_provider == 'private': mean += 0.5
            if person.area_type == 'rural': mean -= 1.0
            elif person.area_type == 'urban': mean += 0.3
            return mean, std
        return self._generate_normal_features(Config.QUALITY_INDICATORS, adjust_quality_score)
    
    def _generate_performance_metrics(self, person: Person) -> Dict:
        def adjust_lambda(metric: str, base_lambda: float) -> float:
            if person.healthcare_provider == 'public': base_lambda *= 1.3
            if person.area_type == 'rural': base_lambda *= 1.5
            return base_lambda
        
        metrics = {}
        for metric, params in Config.PERFORMANCE_METRICS.items():
            adj_lambda = adjust_lambda(metric, params['lambda'])
            value = np.random.poisson(adj_lambda)
            metrics[metric] = value
        
        if person.high_healthcare_utilizer:
            metrics['missed_appointments_count'] *= 2
            metrics['provider_changes_count'] += 1
            
        return metrics

    def _generate_care_access_features(self, person: Person, attempt_date: date) -> Dict:
        def adjust_care_access_prob(setting: str, base_prob: float) -> float:
            if person.healthcare_provider == 'private' and setting in ['specialist_psychiatric', 'intensive_outpatient']:
                return base_prob * 1.4
            if setting == 'emergency_services':
                return base_prob * 2.0
            return base_prob
        
        features = self._generate_weighted_features(
            {f'accessed_{k}': v for k, v in Config.CARE_SETTINGS.items()}, 
            adjust_care_access_prob
        )
        
        # Other access features
        features.update({
            'travel_distance_to_care_km': self._generate_travel_distance(person),
            'transportation_barriers': np.random.random() < 0.25,
            'same_day_access_available': np.random.random() < 0.30,
            'after_hours_access_available': np.random.random() < 0.40,
            'telehealth_available': np.random.random() < 0.70,
            'language_barrier_present': (person.nationality != 'Uruguayan' and np.random.random() < 0.60),
            'cultural_competency_rating': round(max(0, min(10, np.random.normal(7.0, 1.5))), 1),
        })
        
        return features

    def _generate_travel_distance(self, person: Person) -> float:
        area_type = person.area_type
        if area_type == 'urban': distance = np.random.gamma(2, 3)
        elif area_type == 'suburban': distance = np.random.gamma(3, 5)
        else: distance = np.random.gamma(4, 10)
        return round(distance, 1)

    def _generate_coverage_features(self, person: Person) -> Dict:
        features = {
            'primary_insurance': 'private_insurance' if person.healthcare_provider == 'private' else 'public_snis',
            'mental_health_coverage_adequate': np.random.random() < 0.70,
            'medication_coverage_adequate': np.random.random() < 0.80,
            'therapy_sessions_covered': np.random.randint(8, 25)
        }
        
        financial_barrier_prob = 0.5 - (person.socioeconomic_status_score * 0.4) if person.socioeconomic_status_score is not None else 0.3
        
        features.update({
            'financial_barriers_to_care': np.random.random() < financial_barrier_prob,
            'medication_cost_burden': np.random.random() < (financial_barrier_prob * 0.8),
        })
        
        features['delayed_care_due_to_cost'] = features['financial_barriers_to_care'] and np.random.random() < 0.60
        features['skipped_medications_due_to_cost'] = features['medication_cost_burden'] and np.random.random() < 0.40
        
        return features

    def _generate_data_quality_features(self, person: Person) -> Dict:
        """Generate data quality assessment features."""
        base_quality = 0.8
        
        # Adjust based on demographics
        if person.area_type == 'rural':
            base_quality -= 0.2
        if person.socioeconomic_status_score and person.socioeconomic_status_score < 0.3:
            base_quality -= 0.15
        if person.age > 70:
            base_quality -= 0.1
        
        data_quality_score = max(0.0, min(1.0, base_quality + np.random.normal(0, 0.1)))
        
        return {
            'data_quality_score': round(data_quality_score, 3),
            'ehr_integrated': np.random.random() < 0.85,
            'care_plan_documented': np.random.random() < 0.75,
            'crisis_plan_available': np.random.random() < 0.60,
            'multidisciplinary_team': np.random.random() < 0.70,
            'case_manager_assigned': np.random.random() < 0.45,
            'peer_support_available': np.random.random() < 0.55,
            'provider_communication_quality': round(max(1, min(10, np.random.normal(7.5, 1.5))), 1),
            'discharge_planning_quality': round(max(1, min(10, np.random.normal(6.8, 2.0))), 1)
        }


@auto_feature_names
class MachineLearningGenerator(BaseFeatureGenerator):
    """Enhanced machine learning and model development features."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'encoding_schemes': Config.ML_CONFIGS['encoding_schemes'],
            'scaling_methods': Config.ML_CONFIGS['scaling_methods'],
            'imputation_strategies': Config.ML_CONFIGS['imputation_strategies'],
            'cv_strategies': Config.ML_CONFIGS['cv_strategies'],
            'n_folds_options': Config.ML_CONFIGS['n_folds_options'],
            'model_families': Config.ML_CONFIGS['model_families'],
        }

    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        # Data splitting features
        split_features = self._generate_data_splits(person, attempt_date)
        
        # Feature engineering metadata
        engineering_features = self._generate_feature_engineering_metadata()
        
        # Model training features
        training_features = self._generate_training_metadata()
        
        # Prediction and scoring features
        prediction_features = self._generate_prediction_features(person)
        
        # Model interpretability features
        interpretability_features = self._generate_interpretability_features(person)

        features = {
            **split_features, **engineering_features, **training_features,
            **prediction_features, **interpretability_features
        }
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features

    def _generate_data_splits(self, person: Person, attempt_date: date) -> Dict:
        features = {}
        year = attempt_date.year
        features['temporal_split'] = 'train' if year <= 2024 else 'validation' if year == 2025 else 'test'
        
        person_hash = int(hashlib.md5(person.person_id.encode()).hexdigest(), 16)
        np.random.seed(person_hash % 1000)
        features['random_split'] = np.random.choice(['train', 'validation', 'test'], p=[0.7, 0.15, 0.15])
        
        risk_category = person.overall_risk_category
        if risk_category:
            p = [0.6, 0.2, 0.2] if risk_category == 'high' else [0.75, 0.125, 0.125]
            features['stratified_split'] = np.random.choice(['train', 'validation', 'test'], p=p)
        else:
            features['stratified_split'] = features['random_split']
            
        cluster_hash = hash(person.geographic_cluster_id) % 10 if person.geographic_cluster_id else 0
        features['geographic_split'] = 'train' if cluster_hash < 7 else 'validation' if cluster_hash < 9 else 'test'
        
        features['cv_fold'] = person_hash % 5
        features['is_train_set'] = features['temporal_split'] == 'train'
        features['is_validation_set'] = features['temporal_split'] == 'validation'
        features['is_test_set'] = features['temporal_split'] == 'test'
        return features

    def _generate_feature_engineering_metadata(self) -> Dict:
        return {
            'categorical_encoding_strategy': np.random.choice(Config.ML_CONFIGS['encoding_schemes']),
            'numerical_scaling_method': np.random.choice(Config.ML_CONFIGS['scaling_methods']),
            'missing_value_strategy': np.random.choice(Config.ML_CONFIGS['imputation_strategies']),
            'high_cardinality_features': np.random.randint(0, 5),
            'sparse_features_count': np.random.randint(0, 10),
            'engineered_features_count': np.random.randint(5, 25),
            'data_quality_flag': np.random.choice(['clean', 'minor_issues', 'major_issues'], p=[0.7, 0.25, 0.05]),
            'outlier_score': round(np.random.beta(2, 8), 3),
            'missing_data_percentage': round(np.random.beta(1, 9) * 20, 1),
            'feature_correlation_max': round(np.random.beta(3, 5), 3),
            'target_correlation': round(np.random.normal(0, 0.3), 3),
            'variance_inflation_factor': round(np.random.gamma(2, 2), 2)
        }
    
    def _generate_training_metadata(self) -> Dict:
        return {
            'primary_model_family': np.random.choice(Config.ML_CONFIGS['model_families']),
            'ensemble_method': np.random.choice(['voting', 'stacking', 'blending', 'none'], p=[0.25, 0.25, 0.15, 0.35]),
            'cv_strategy': np.random.choice(Config.ML_CONFIGS['cv_strategies']),
            'cv_folds': np.random.choice(Config.ML_CONFIGS['n_folds_options']),
            'hyperparameter_tuning_method': np.random.choice(['grid_search', 'random_search', 'bayesian_optimization', 'none'], p=[0.2, 0.3, 0.3, 0.2]),
            'optimization_metric': np.random.choice(['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']),
            'training_time_minutes': round(np.random.gamma(3, 10), 1),
            'convergence_achieved': np.random.random() < 0.85,
            'early_stopping_triggered': np.random.random() < 0.30
        }

    def _generate_prediction_features(self, person: Person) -> Dict:
        features = {}
        prediction_horizons = ['7_day', '30_day', '90_day', '6_month', '1_year']
        base_prob = 0.15
        
        if person.overall_risk_category == 'high': base_prob = 0.45
        elif person.overall_risk_category == 'moderate': base_prob = 0.25
        elif person.overall_risk_category == 'low': base_prob = 0.10
        
        time_decay = {'7_day': 1.0, '30_day': 0.8, '90_day': 0.6, '6_month': 0.4, '1_year': 0.3}
        
        for horizon in prediction_horizons:
            adjusted_prob = base_prob * time_decay[horizon]
            prediction_score = np.random.beta(adjusted_prob * 10, (1 - adjusted_prob) * 10)
            uncertainty = np.random.beta(2, 8) * 0.2
            
            features[f'prediction_score_{horizon}'] = round(prediction_score, 4)
            features[f'prediction_lower_ci_{horizon}'] = round(max(0, prediction_score - uncertainty), 4)
            features[f'prediction_upper_ci_{horizon}'] = round(min(1, prediction_score + uncertainty), 4)

        features['model_consensus_score'] = round(np.random.beta(4, 2), 3)
        features['prediction_stability'] = round(np.random.beta(3, 2), 3)
        features['ensemble_weight'] = round(np.random.beta(5, 2), 3)
        features['calibration_score'] = round(np.random.beta(4, 3), 3)
        features['over_confident'] = features['model_consensus_score'] > 0.9 and features['calibration_score'] < 0.7
        features['under_confident'] = features['model_consensus_score'] < 0.5 and features['calibration_score'] > 0.8
        return features

    def _generate_interpretability_features(self, person: Person) -> Dict:
        features = {}
        important_features = [
            'age_importance', 'sex_importance', 'previous_attempts_importance',
            'mental_health_treatment_importance', 'depression_severity_importance',
            'social_support_importance', 'hopelessness_importance'
        ]
        importance_values = np.random.dirichlet([2, 2, 4, 3, 4, 3, 4])
        for i, feature_name in enumerate(important_features):
            features[feature_name] = round(importance_values[i], 4)
        
        features['local_explanation_confidence'] = round(np.random.beta(3, 2), 3)
        features['counterfactual_distance'] = round(np.random.gamma(2, 0.5), 3)
        features['model_complexity_score'] = round(np.random.beta(3, 3), 3)
        features['linear_separability'] = round(np.random.beta(2, 3), 3)
        features['feature_interaction_strength'] = round(np.random.beta(2, 4), 3)
        
        features['clinical_rule_triggered'] = np.random.choice([
            'high_risk_youth', 'multiple_attempts_history', 'acute_stressor_present',
            'treatment_non_adherence', 'none'
        ], p=[0.2, 0.25, 0.3, 0.15, 0.1])
        
        features['recommendation_type'] = np.random.choice([
            'immediate_intervention', 'enhanced_monitoring', 'standard_follow_up',
            'referral_specialist', 'crisis_team_activation'
        ], p=[0.15, 0.25, 0.35, 0.2, 0.05])

        features['epistemic_uncertainty'] = round(np.random.beta(2, 5), 3)
        features['aleatoric_uncertainty'] = round(np.random.beta(3, 4), 3)
        features['total_uncertainty'] = round(features['epistemic_uncertainty'] + features['aleatoric_uncertainty'], 3)
        
        return features


@auto_feature_names
class BiometricsDigitalGenerator(BaseFeatureGenerator):
    """Enhanced digital biomarkers and wearable device features."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'device_adoption': Config.DEVICE_ADOPTION,
            'biomarker_categories': Config.BIOMARKER_CATEGORIES,
            'app_categories': ['meditation_mindfulness', 'mood_tracking', 'therapy_platforms',
                               'crisis_support', 'peer_support', 'cognitive_training']
        }
    
    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        age = person.age
        device_features = self._generate_device_features(age)
        biomarker_features = self._generate_digital_biomarkers(device_features)
        app_usage_features = self._generate_app_usage_patterns(age)
        passive_features = self._generate_passive_monitoring(device_features)

        features = {
            **device_features,
            **biomarker_features,
            **app_usage_features,
            **passive_features
        }
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features

    def _generate_device_features(self, age: int) -> Dict:
        def adjust_adoption_rate(device: str, base_rate: float) -> float:
            if age < 25: return base_rate * 1.4
            elif age < 40: return base_rate * 1.2
            elif age > 60: return base_rate * 0.6
            return base_rate
        
        features = {}
        device_adoption = self._generate_weighted_features(Config.DEVICE_ADOPTION, adjust_adoption_rate)
        for device, owns_device in device_adoption.items():
            features[f'owns_{device}'] = owns_device
            if owns_device:
                features[f'{device}_daily_usage_hours'] = round(np.random.gamma(2, 2), 1)
                features[f'{device}_engagement_score'] = round(np.random.beta(3, 2), 3)
            else:
                features[f'{device}_daily_usage_hours'] = 0.0
                features[f'{device}_engagement_score'] = 0.0

        total_devices = sum(features[f'owns_{d}'] for d in Config.DEVICE_ADOPTION)
        features['digital_device_count'] = total_devices
        features['high_digital_engagement'] = total_devices >= 3
        return features
    
    def _generate_digital_biomarkers(self, device_features: Dict) -> Dict:
        features = {}
        
        # Sleep metrics (require sleep tracker or smartwatch)
        if device_features.get('owns_sleep_tracker') or device_features.get('owns_smartwatch'):
            features.update({
                'avg_sleep_duration_hours': round(np.random.normal(6.8, 1.2), 1),
                'sleep_efficiency_percent': round(np.random.normal(82, 12), 1),
                'sleep_latency_minutes': round(np.random.gamma(2, 15), 1),
                'rem_sleep_percent': round(np.random.normal(22, 5), 1),
                'sleep_regularity_score': round(np.random.beta(3, 2), 3),
                'sleep_disruption_events': np.random.poisson(2)
            })
        
        # Activity metrics (require fitness tracker or smartwatch)
        if device_features.get('owns_fitness_tracker') or device_features.get('owns_smartwatch'):
            features.update({
                'daily_steps_avg': int(np.random.gamma(3, 2000)),
                'active_minutes_daily': round(np.random.gamma(3, 15), 1),
                'sedentary_time_hours': round(np.random.normal(8.5, 2.0), 1),
                'heart_rate_variability': round(np.random.normal(35, 10), 1),
                'resting_heart_rate': int(np.random.normal(72, 12)),
                'activity_consistency_score': round(np.random.beta(2, 3), 3)
            })
        
        # Digital behavior metrics (require smartphone)
        if device_features.get('owns_smartphone'):
            features.update({
                'screen_time_hours_daily': round(np.random.gamma(4, 1.5), 1),
                'app_switches_daily': np.random.poisson(120),
                'typing_speed_wpm': round(np.random.normal(35, 8), 1),
                'typing_irregularity': round(np.random.beta(2, 4), 3),
                'call_duration_avg_minutes': round(np.random.gamma(2, 3), 1),
                'social_app_usage_hours': round(np.random.gamma(2, 1), 1),
                'location_entropy': round(np.random.beta(3, 3), 3)
            })
        
        return features

    def _generate_app_usage_patterns(self, age: int) -> Dict:
        features = {}
        app_categories = ['meditation_mindfulness', 'mood_tracking', 'therapy_platforms',
                          'crisis_support', 'peer_support', 'cognitive_training']

        for app_type in app_categories:
            base_rate = {
                'meditation_mindfulness': 0.25, 'mood_tracking': 0.15, 'therapy_platforms': 0.10,
                'crisis_support': 0.08, 'peer_support': 0.12, 'cognitive_training': 0.08
            }.get(app_type, 0.1)
            
            adjusted_rate = base_rate * (1.5 if age < 30 else 1.0 if age < 50 else 0.7)
            uses_app = np.random.random() < adjusted_rate
            
            features[f'uses_{app_type}_app'] = uses_app
            if uses_app:
                features[f'{app_type}_sessions_per_week'] = np.random.poisson(4)
                features[f'{app_type}_avg_session_minutes'] = round(np.random.gamma(2, 8), 1)
                features[f'{app_type}_adherence_rate'] = round(np.random.beta(3, 2), 3)
            else:
                features[f'{app_type}_sessions_per_week'] = 0
                features[f'{app_type}_avg_session_minutes'] = 0.0
                features[f'{app_type}_adherence_rate'] = 0.0

        total_apps = sum(features[f'uses_{app}_app'] for app in app_categories)
        features['mental_health_app_count'] = total_apps
        features['digital_therapy_engaged'] = total_apps >= 2
        return features

    def _generate_passive_monitoring(self, device_features: Dict) -> Dict:
        features = {}
        
        # Communication patterns (require smartphone)
        if device_features.get('owns_smartphone'):
            features.update({
                'text_message_frequency_daily': np.random.poisson(25),
                'call_frequency_daily': np.random.poisson(3),
                'social_contact_diversity': np.random.poisson(8),
                'response_time_minutes_avg': round(np.random.gamma(3, 20), 1),
                'communication_regularity': round(np.random.beta(3, 2), 3)
            })
        
        features['notification_response_rate'] = round(np.random.beta(4, 3), 3)
        features['app_usage_fragmentation'] = round(np.random.beta(2, 3), 3)
        features['digital_routine_consistency'] = round(np.random.beta(3, 2), 3)

        # Environmental sensors (subset of users have these)
        if np.random.random() < 0.30:
            features.update({
                'ambient_light_avg_lux': int(np.random.gamma(3, 100)),
                'ambient_noise_avg_db': round(np.random.normal(45, 10), 1),
                'temperature_avg_celsius': round(np.random.normal(22, 3), 1),
                'air_quality_index': int(np.random.gamma(2, 25))
            })
        
        # Calculate completeness score
        valid_biomarkers = sum(1 for k, v in features.items() 
                              if v is not None and any(metric in k for metric in 
                                 ['score', 'count', 'avg', 'percent', 'minutes', 'wpm', 'db', 'lux', 'index']))
        features['digital_biomarker_completeness'] = round(valid_biomarkers / 25, 2)
        return features


class DataValidator:
    """Enhanced validation framework for generated data."""
    
    @staticmethod
    def validate_generated_data(dataset: pd.DataFrame) -> Dict[str, bool]:
        """Comprehensive data validation with detailed checks."""
        checks = {}
        
        # Basic data integrity
        checks['no_null_person_ids'] = dataset['person_id'].notna().all()
        checks['valid_age_range'] = dataset['age_at_attempt'].between(10, 100).all()
        checks['valid_sex_values'] = dataset['sex'].isin(['M', 'F']).all()
        
        # Clinical coherence
        checks['age_method_consistency'] = DataValidator._check_age_method_patterns(dataset)
        checks['clinical_coherence'] = DataValidator._check_clinical_combinations(dataset)
        checks['temporal_logic'] = DataValidator._check_temporal_sequences(dataset)
        
        # Statistical distributions
        checks['statistical_distributions'] = DataValidator._check_marginal_distributions(dataset)
        
        # Missing data patterns
        checks['realistic_missingness'] = DataValidator._check_missing_patterns(dataset)
        
        return checks

    @staticmethod
    def _check_age_method_patterns(dataset: pd.DataFrame) -> bool:
        """Check if age-method combinations are realistic."""
        try:
            # Certain methods are rare in very young populations
            young_firearm = dataset[(dataset['age_at_attempt'] < 16) & 
                                  (dataset['method_primary'] == 'Firearm')]
            return len(young_firearm) / len(dataset) < 0.02
        except KeyError:
            return True

    @staticmethod
    def _check_clinical_combinations(dataset: pd.DataFrame) -> bool:
        """Check for realistic clinical feature combinations."""
        try:
            # People with diabetes should be more likely to have other conditions
            if 'has_diabetes' in dataset.columns and 'has_hypertension' in dataset.columns:
                diabetes_hypertension_rate = dataset[dataset['has_diabetes']]['has_hypertension'].mean()
                general_hypertension_rate = dataset['has_hypertension'].mean()
                return diabetes_hypertension_rate > general_hypertension_rate
            return True
        except KeyError:
            return True

    @staticmethod
    def _check_temporal_sequences(dataset: pd.DataFrame) -> bool:
        """Check temporal logic in the data."""
        try:
            # Attempt dates should be within expected range
            min_date = dataset['attempt_date'].min()
            max_date = dataset['attempt_date'].max()
            return (min_date >= date(2020, 1, 1) and max_date <= date(2030, 12, 31))
        except (KeyError, TypeError):
            return True

    @staticmethod
    def _check_marginal_distributions(dataset: pd.DataFrame) -> bool:
        """Check if marginal distributions are realistic."""
        try:
            # Check sex distribution
            if 'sex' in dataset.columns:
                female_prop = (dataset['sex'] == 'F').mean()
                return 0.3 < female_prop < 0.7
            return True
        except KeyError:
            return True

    @staticmethod
    def _check_missing_patterns(dataset: pd.DataFrame) -> bool:
        """Check if missing data patterns are realistic."""
        try:
            # Digital features should have more missing data in rural areas
            digital_cols = [col for col in dataset.columns if 'digital_' in col.lower()]
            if digital_cols and 'area_type' in dataset.columns:
                rural_missing = dataset[dataset['area_type'] == 'rural'][digital_cols].isnull().mean().mean()
                urban_missing = dataset[dataset['area_type'] == 'urban'][digital_cols].isnull().mean().mean()
                return rural_missing >= urban_missing
            return True
        except KeyError:
            return True


class BaseSurveillanceDataGenerator(ABC):
    """Enhanced base class for surveillance data generation."""
    
    def __init__(self, start_year: int, end_year: int, random_seed: int, 
                 realism_config: Optional[RealismConfig] = None):
        self.start_year = start_year
        self.end_year = end_year
        self.random_seed = random_seed
        self.realism_config = realism_config or RealismConfig()
        self.feature_generators: List[BaseFeatureGenerator] = []
        self.person_history: Dict[str, Person] = {}  # Track person history
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def add_feature_generator(self, generator: BaseFeatureGenerator):
        """Add a feature generator with realism config."""
        generator.realism_config = self.realism_config
        self.feature_generators.append(generator)

    def generate_person_id(self, index: int) -> str:
        return f"PERSON-{str(index).zfill(6)}"
        
    def generate_person_data(self, person_id: str, year: int) -> Person:
        """Enhanced person generation with history tracking."""
        # Check if person exists in history
        if person_id in self.person_history:
            existing_person = self.person_history[person_id]
            # Update age but keep other stable characteristics
            existing_person.age += (year - existing_person.additional_data.get('last_year', year))
            existing_person.additional_data['last_year'] = year
            return existing_person
        
        # Create new person
        age = random.randint(15, 85)
        sex = random.choice(['M', 'F'])
        person = Person(
            person_id=person_id,
            age=age,
            sex=sex,
            nationality=random.choice(['Uruguayan', 'Foreign']),
            healthcare_provider=random.choice(['public', 'private']),
            additional_data={'last_year': year}
        )
        
        self.person_history[person_id] = person
        return person

    def generate_complete_dataset(self, n_records: int = 1000) -> pd.DataFrame:
        """Enhanced dataset generation with validation."""
        data = []
        
        print(f"Generating {n_records} records...")
        
        for i in range(n_records):
            if i % 100 == 0:
                print(f"Progress: {i}/{n_records}")
                
            person_id = self.generate_person_id(i % 500)  # Allow repeat individuals
            year = random.randint(self.start_year, self.end_year)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            attempt_date = date(year, month, day)

            person = self.generate_person_data(person_id, year)
            
            features = {
                'person_id': person_id,
                'age_at_attempt': person.age,
                'sex': person.sex,
                'attempt_date': attempt_date,
                'method_primary': random.choice(['Overdose', 'Suffocation', 'Firearm', 'Cutting']),
                'followup_status': random.choice(['Complete', 'Incomplete', 'Refused']),
            }

            # Generate features from all generators
            for generator in self.feature_generators:
                try:
                    generated_features = generator.generate_features(person, attempt_date, {})
                    features.update(generated_features)
                    
                    # Update person's historical features for consistency
                    person.historical_features.update(generated_features)
                    
                except Exception as e:
                    print(f"Warning: Error in {generator.__class__.__name__}: {e}")
                    continue
                    
            data.append(features)

        dataset = pd.DataFrame(data)
        
        # Validate the generated data
        validation_results = DataValidator.validate_generated_data(dataset)
        failed_checks = [check for check, passed in validation_results.items() if not passed]
        
        if failed_checks:
            print(f"Warning: Failed validation checks: {failed_checks}")
        else:
            print("All validation checks passed!")
            
        return dataset


class IndustryStandardSurveillanceGenerator(BaseSurveillanceDataGenerator):
    """Enhanced surveillance generator with comprehensive industry-standard features."""
    
    def __init__(self, start_year: int = 2023, end_year: int = 2027,
                 random_seed: int = 42, feature_set: str = 'comprehensive',
                 realism_config: Optional[RealismConfig] = None):
        super().__init__(start_year, end_year, random_seed, realism_config)
        self.feature_set = feature_set
        self._initialize_industry_generators()

    def _initialize_industry_generators(self):
        """Initialize generators based on feature set selection."""
        if self.feature_set in ['clinical', 'research', 'comprehensive']:
            self.add_feature_generator(SociodemographicGenerator(self.realism_config))
            self.add_feature_generator(ClinicalComorbidityGenerator(self.realism_config))
            self.add_feature_generator(RiskFactorGenerator(self.realism_config))
        
        if self.feature_set in ['research', 'comprehensive']:
            self.add_feature_generator(HealthcareSystemGenerator(self.realism_config))
            self.add_feature_generator(MachineLearningGenerator(self.realism_config))
        
        if self.feature_set == 'comprehensive':
            self.add_feature_generator(BiometricsDigitalGenerator(self.realism_config))

    def generate_comprehensive_dataset(self, n_records: int = 1000) -> pd.DataFrame:
        """Generate comprehensive dataset with enhanced reporting."""
        print(f"Generating comprehensive dataset with {self.feature_set} feature set...")
        print(f"Realism settings: correlation={self.realism_config.correlation_strength}, "
              f"noise={self.realism_config.noise_level}, missing={self.realism_config.missing_data_rate}")
        
        dataset = self.generate_complete_dataset(n_records)
        dataset['feature_set_type'] = self.feature_set
        dataset['generation_timestamp'] = datetime.now()
        
        # Enhanced reporting
        feature_counts = self._calculate_feature_counts(dataset)
        quality_metrics = self._calculate_quality_metrics(dataset)
        
        print("\n" + "="*60)
        print("DATASET GENERATION SUMMARY")
        print("="*60)
        print(f"Total records: {len(dataset):,}")
        print(f"Total features: {len(dataset.columns)}")
        print(f"Date range: {dataset['attempt_date'].min()} to {dataset['attempt_date'].max()}")
        print(f"Unique individuals: {dataset['person_id'].nunique()}")
        
        print("\nFeature Categories:")
        for category, count in feature_counts.items():
            print(f"  {category}: {count} features")
        
        print("\nData Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value}")
        
        return dataset

    def _calculate_feature_counts(self, dataset: pd.DataFrame) -> Dict[str, int]:
        """Enhanced feature counting with better categorization."""
        feature_counts = {}
        columns = dataset.columns.tolist()
        
        feature_counts['Core Surveillance'] = len([c for c in columns if any(k in c.lower() 
            for k in ['attempt_date', 'method', 'age_at_attempt', 'sex', 'followup', 'person_id'])])
        
        feature_counts['Sociodemographic'] = len([c for c in columns if any(k in c.lower() 
            for k in ['education', 'employment', 'marital', 'household', 'socioeconomic', 'area_type'])])
        
        feature_counts['Clinical'] = len([c for c in columns if any(k in c.lower() 
            for k in ['has_', 'prescribed_', 'gaf_', 'clinical_', 'comorbidity', 'medication', 'severity'])])
        
        feature_counts['Risk Factors'] = len([c for c in columns if any(k in c.lower() 
            for k in ['risk_', 'protective_', 'hopelessness', 'depression_severity', 'trauma', 'ace', 'stress'])])
        
        feature_counts['Healthcare System'] = len([c for c in columns if any(k in c.lower() 
            for k in ['care_', 'provider_', 'insurance_', 'accessed_', 'quality_', 'wait_time', 'coverage'])])
        
        feature_counts['Machine Learning'] = len([c for c in columns if any(k in c.lower() 
            for k in ['split', 'prediction_', 'model_', 'importance', 'cv_fold', 'tuning', 'encoding', 'uncertainty'])])
        
        feature_counts['Digital Biomarkers'] = len([c for c in columns if any(k in c.lower() 
            for k in ['owns_', 'digital_', 'app_', 'sleep_', 'biomarker', 'screen_time', 'steps'])])
        
        return feature_counts

    def _calculate_quality_metrics(self, dataset: pd.DataFrame) -> Dict[str, str]:
        """Calculate and format data quality metrics."""
        metrics = {}
        
        # Missing data percentage
        missing_pct = (dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))) * 100
        metrics['Missing data'] = f"{missing_pct:.1f}%"
        
        # Duplicate records
        duplicate_pct = (dataset.duplicated().sum() / len(dataset)) * 100
        metrics['Duplicate records'] = f"{duplicate_pct:.1f}%"
        
        # Categorical feature variety
        categorical_cols = dataset.select_dtypes(include=['object']).columns
        avg_categories = dataset[categorical_cols].nunique().mean() if len(categorical_cols) > 0 else 0
        metrics['Avg categories per categorical feature'] = f"{avg_categories:.1f}"
        
        # Numerical feature ranges
        numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        infinite_values = dataset[numerical_cols].isin([np.inf, -np.inf]).sum().sum()
        metrics['Infinite values'] = str(infinite_values)
        
        return metrics

    def export_feature_documentation(self, output_path: str = "feature_documentation.json"):
        """Export comprehensive feature documentation."""
        documentation = {
            'generation_config': {
                'feature_set': self.feature_set,
                'start_year': self.start_year,
                'end_year': self.end_year,
                'realism_settings': {
                    'correlation_strength': self.realism_config.correlation_strength,
                    'noise_level': self.realism_config.noise_level,
                    'missing_data_rate': self.realism_config.missing_data_rate,
                    'outlier_rate': self.realism_config.outlier_rate
                }
            },
            'feature_generators': {}
        }
        
        for generator in self.feature_generators:
            generator_name = generator.__class__.__name__
            documentation['feature_generators'][generator_name] = {
                'feature_names': generator.get_feature_names(),
                'feature_count': len(generator.get_feature_names()),
                'distributions': getattr(generator, '_distributions', {})
            }
        
        with open(output_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        print(f"Feature documentation exported to {output_path}")


def demonstrate_enhanced_generators():
    """Comprehensive demonstration of enhanced generators."""
    print("="*80)
    print("ENHANCED INDUSTRY-STANDARD FEATURE GENERATORS DEMONSTRATION")
    print("="*80)
    
    # Create different realism configurations
    configs = {
        'high_realism': RealismConfig(
            correlation_strength=0.8,
            noise_level=0.05,
            missing_data_rate=0.03,
            temporal_consistency=True,
            seasonal_effects=True,
            covid_impact=True
        ),
        'moderate_realism': RealismConfig(
            correlation_strength=0.6,
            noise_level=0.1,
            missing_data_rate=0.08,
            temporal_consistency=True,
            seasonal_effects=False,
            covid_impact=False
        ),
        'basic': RealismConfig(
            correlation_strength=0.4,
            noise_level=0.15,
            missing_data_rate=0.15,
            temporal_consistency=False,
            seasonal_effects=False,
            covid_impact=False
        )
    }
    
    print("Testing different realism configurations...")
    
    for config_name, config in configs.items():
        print(f"\n{'-'*60}")
        print(f"CONFIGURATION: {config_name.upper()}")
        print(f"{'-'*60}")
        
        generator = IndustryStandardSurveillanceGenerator(
            start_year=2023,
            end_year=2025,
            random_seed=42,
            feature_set='comprehensive',
            realism_config=config
        )
        
        print(f"Initialized generator with {len(generator.feature_generators)} feature generators:")
        for i, gen in enumerate(generator.feature_generators):
            print(f"  {i+1}. {gen.__class__.__name__}")
        
        # Generate smaller sample for demonstration
        print(f"\nGenerating sample dataset with {config_name} realism...")
        dataset = generator.generate_comprehensive_dataset(n_records=200)
        
        # Show sample of key features
        key_features = [
            'person_id', 'age_at_attempt', 'sex', 'method_primary',
            'education_level', 'employment_status', 'overall_risk_category'
        ]
        
        # Add features that exist in dataset
        optional_features = [
            'has_major_depression', 'prediction_score_30_day', 'owns_smartphone',
            'covid_impact_factor', 'seasonal_depression_risk', 'data_quality_score'
        ]
        
        available_features = key_features + [f for f in optional_features if f in dataset.columns]
        
        print(f"\nSample of key features ({config_name}):")
        print(dataset[available_features].head())
        
        # Validation results
        validation_results = DataValidator.validate_generated_data(dataset)
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        
        print(f"\nValidation: {passed_checks}/{total_checks} checks passed")
        
        # Export documentation for high realism config
        if config_name == 'high_realism':
            try:
                generator.export_feature_documentation("enhanced_feature_docs.json")
            except Exception as e:
                print(f"Note: Could not export documentation: {e}")
    
    print(f"\n{'='*80}")
    print("ENHANCEMENT SUMMARY")
    print("="*80)
    print("✅ Added missing configuration dependencies")
    print("✅ Implemented temporal consistency tracking")
    print("✅ Added COVID-19 and seasonal effects")
    print("✅ Enhanced missing data patterns")
    print("✅ Implemented performance caching")
    print("✅ Added comprehensive validation framework")
    print("✅ Improved chronic condition stability")
    print("✅ Enhanced medication adherence modeling")
    print("✅ Added data quality assessment features")
    print("✅ Implemented configurable realism levels")
    print("✅ Added feature documentation export")
    
    return dataset


def create_specialized_datasets():
    """Create specialized datasets for different use cases."""
    print("\n" + "="*80)
    print("CREATING SPECIALIZED DATASETS")
    print("="*80)
    
    # High-risk population focused dataset
    high_risk_config = RealismConfig(
        correlation_strength=0.9,  # Strong correlations in high-risk
        noise_level=0.03,
        missing_data_rate=0.02,    # Better data quality for high-risk
        temporal_consistency=True,
        seasonal_effects=True,
        covid_impact=True
    )
    
    # Research quality dataset
    research_config = RealismConfig(
        correlation_strength=0.75,
        noise_level=0.05,
        missing_data_rate=0.01,    # Minimal missing data
        temporal_consistency=True,
        seasonal_effects=True,
        covid_impact=True,
        data_quality_variation=False  # Consistent quality
    )
    
    # Real-world clinical dataset (more realistic)
    clinical_config = RealismConfig(
        correlation_strength=0.6,
        noise_level=0.12,
        missing_data_rate=0.15,    # Realistic missing data
        outlier_rate=0.05,         # Some outliers
        temporal_consistency=True,
        seasonal_effects=True,
        covid_impact=True,
        data_quality_variation=True
    )
    
    datasets = {}
    
    for name, config, feature_set in [
        ("high_risk_research", high_risk_config, "research"),
        ("research_quality", research_config, "comprehensive"), 
        ("clinical_realistic", clinical_config, "clinical")
    ]:
        print(f"\nGenerating {name} dataset...")
        
        generator = IndustryStandardSurveillanceGenerator(
            start_year=2020,  # Include COVID period
            end_year=2025,
            random_seed=42,
            feature_set=feature_set,
            realism_config=config
        )
        
        dataset = generator.generate_comprehensive_dataset(n_records=300)
        datasets[name] = dataset
        
        print(f"{name}: {len(dataset)} records, {len(dataset.columns)} features")
    
    return datasets


# Performance benchmarking
def benchmark_performance():
    """Benchmark the performance of enhanced generators."""
    import time
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKING")
    print("="*80)
    
    # Test different record counts
    record_counts = [100, 500, 1000]
    
    generator = IndustryStandardSurveillanceGenerator(
        start_year=2023,
        end_year=2025,
        random_seed=42,
        feature_set='comprehensive',
        realism_config=RealismConfig()
    )
    
    for n_records in record_counts:
        print(f"\nBenchmarking {n_records} records...")
        
        start_time = time.time()
        dataset = generator.generate_complete_dataset(n_records)
        end_time = time.time()
        
        duration = end_time - start_time
        records_per_second = n_records / duration
        
        print(f"  Time: {duration:.2f} seconds")
        print(f"  Speed: {records_per_second:.1f} records/second")
        print(f"  Memory: {dataset.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    # Run comprehensive demonstration
    sample_dataset = demonstrate_enhanced_generators()
    
    # Create specialized datasets
    specialized_datasets = create_specialized_datasets()
    
    # Benchmark performance
    benchmark_performance()
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("The enhanced synthetic data generator includes:")
    print("• Realistic temporal trends and consistency")
    print("• Configurable missing data patterns")
    print("• COVID-19 and seasonal effects")
    print("• Comprehensive validation framework")
    print("• Performance optimizations")
    print("• Multiple realism configurations")
    print("• Specialized dataset generation")
    print("• Enhanced documentation and reporting")
    print("="*80)weighted_features(self, distributions: Dict[str, float], 
                                  adjustment_func: Optional[Callable] = None) -> Dict[str, Any]:
        """Generates features from a weighted distribution with optional adjustments."""
        features = {}
        for key, base_rate in distributions.items():
            adjusted_rate = adjustment_func(key, base_rate) if adjustment_func else base_rate
            # Add noise based on realism config
            noise = np.random.normal(0, self.realism_config.noise_level)
            adjusted_rate = max(0.0, min(1.0, adjusted_rate + noise))
            features[key] = np.random.random() < adjusted_rate
        return features
    
    def _generate_normal_features(self, distributions: Dict[str, Dict[str, float]], 
                                adjustment_func: Optional[Callable] = None) -> Dict[str, float]:
        """Enhanced normal feature generation with outlier handling."""
        features = {}
        for key, params in distributions.items():
            mean, std = params['mean'], params['std']
            min_val, max_val = params['range']
            
            if adjustment_func:
                mean, std = adjustment_func(key, mean, std)
            
            # Generate outliers based on realism config
            if np.random.random() < self.realism_config.outlier_rate:
                score = np.random.uniform(min_val, max_val)  # Random outlier
            else:
                score = np.random.normal(mean, std)
                score = max(min_val, min(max_val, score))
            
            features[key] = round(score, 1)
        return features

    @lru_cache(maxsize=1000)
    def _cached_risk_calculation(self, age: int, sex: str, conditions_hash: int) -> float:
        """Cache expensive risk calculations for performance."""
        # This would contain the actual risk calculation logic
        base_risk = 0.1
        if age < 25 or age > 65:
            base_risk *= 1.2
        if sex == 'F':
            base_risk *= 1.1
        return base_risk

    def _add_temporal_trends(self, features: Dict, attempt_date: date, person: Person) -> Dict:
        """Add time-based trends including COVID impact and seasonal patterns."""
        if not self.realism_config.seasonal_effects and not self.realism_config.covid_impact:
            return features
        
        # COVID impact
        if self.realism_config.covid_impact:
            covid_factor = self._calculate_covid_impact(attempt_date)
            if covid_factor > 1.0:
                features['covid_impact_factor'] = round(covid_factor, 2)
                # Adjust specific features
                if 'depression_severity_phq9' in features:
                    features['depression_severity_phq9'] *= covid_factor
                if 'anxiety_severity_gad7' in features:
                    features['anxiety_severity_gad7'] *= covid_factor
                if 'social_isolation' in features:
                    features['social_isolation'] = features['social_isolation'] or (np.random.random() < 0.3)
        
        # Seasonal patterns
        if self.realism_config.seasonal_effects:
            month = attempt_date.month
            for feature, monthly_adjustments in Config.SEASONAL_ADJUSTMENTS.items():
                if feature in features and month in monthly_adjustments:
                    adjustment = monthly_adjustments[month]
                    if isinstance(features[feature], (int, float)):
                        features[feature] *= adjustment
                    features['seasonal_depression_risk'] = adjustment
        
        return features

    def _calculate_covid_impact(self, attempt_date: date) -> float:
        """Calculate COVID impact factor based on date."""
        covid_factor = 1.0
        
        for period, (start_date, end_date) in Config.COVID_PERIODS.items():
            if start_date <= attempt_date <= end_date:
                if period == 'initial_lockdown':
                    covid_factor = np.random.normal(1.4, 0.2)
                elif period == 'second_wave':
                    covid_factor = np.random.normal(1.3, 0.15)
                elif period == 'recovery_phase':
                    covid_factor = np.random.normal(1.1, 0.1)
                break
        
        return max(1.0, covid_factor)

    def _introduce_realistic_missingness(self, features: Dict, person: Person) -> Dict:
        """Simulate realistic missing data patterns based on demographics."""
        if not self.realism_config.data_quality_variation:
            return features
        
        missing_prob = self.realism_config.missing_data_rate
        
        # Adjust missing probability based on demographics
        if person.area_type == 'rural':
            missing_prob *= 2.0  # Rural areas have more missing data
        if person.socioeconomic_status_score and person.socioeconomic_status_score < 0.3:
            missing_prob *= 1.5  # Lower SES = more missing data
        if person.age > 70:
            missing_prob *= 1.3  # Elderly have more missing digital data
        
        # Apply missingness to specific feature categories
        digital_features = [k for k in features if any(term in k.lower() for term in 
                           ['digital_', 'app_', 'smartphone', 'biomarker', 'screen_time'])]
        
        for feature in digital_features:
            if np.random.random() < missing_prob:
                features[feature] = None
        
        return features

    def _ensure_person_consistency(self, features: Dict, person: Person) -> Dict:
        """Maintain consistency across multiple records for same person."""
        if not self.realism_config.temporal_consistency or not person.historical_features:
            return features
        
        # Chronic conditions don't disappear
        for condition in Config.CHRONIC_CONDITIONS:
            condition_key = f'has_{condition}'
            if condition_key in person.historical_features and person.historical_features[condition_key]:
                features[condition_key] = True
        
        # Some demographic features are stable
        stable_features = ['education_level', 'ace_score', 'childhood_trauma_history']
        for feature in stable_features:
            if feature in person.historical_features:
                features[feature] = person.historical_features[feature]
        
        return features

    @abstractmethod
    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        """Abstract method to generate a dictionary of features."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Abstract method to get the list of feature names."""
        pass

@auto_feature_names
class SociodemographicGenerator(BaseFeatureGenerator):
    """Enhanced industry-standard sociodemographic features."""
    
    def __init__(self, realism_config: Optional[RealismConfig] = None):
        super().__init__(realism_config)
        self._distributions = {
            'education_levels': Config.EDUCATION_LEVELS,
            'employment_status': Config.EMPLOYMENT_STATUS,
            'marital_status': Config.MARITAL_STATUS,
            'area_classification': Config.AREA_CLASSIFICATION,
        }

    def generate_features(self, person: Person, attempt_date: date, context: Dict) -> Dict:
        age = person.age
        sex = person.sex

        # Education level (age-dependent)
        education = self._generate_education_level(age)
        
        # Employment status (age and education dependent)
        employment = self._generate_employment_status(age, education)
        
        # Marital status (age and sex dependent)
        marital = self._generate_marital_status(age, sex)
        
        # Generate other features
        household_size = self._generate_household_size(age, marital)
        children_in_household = self._generate_children_count(age, marital)
        area_type = np.random.choice(list(Config.AREA_CLASSIFICATION.keys()), 
                                   p=list(Config.AREA_CLASSIFICATION.values()))
        
        # Update person object with generated features for downstream dependencies
        person.area_type = area_type
        person.marital_status = marital
        person.employment_status = employment

        # Socioeconomic status proxy (composite of education, employment, area)
        ses_score = self._calculate_ses_score(education, employment, area_type)
        person.socioeconomic_status_score = ses_score
        
        # Social support indicators
        social_support_score = self._generate_social_support_score(marital, household_size)

        features = {
            'education_level': education,
            'employment_status': employment,
            'marital_status': marital,
            'household_size': household_size,
            'children_in_household': children_in_household,
            'area_type': area_type,
            'socioeconomic_status_score': round(ses_score, 3),
            'social_support_score': round(social_support_score, 3),
            'lives_alone': household_size == 1,
            'primary_earner': self._is_primary_earner(employment, marital, sex)
        }
        
        # Apply enhancements
        features = self._ensure_person_consistency(features, person)
        features = self._add_temporal_trends(features, attempt_date, person)
        features = self._introduce_realistic_missingness(features, person)
        
        return features
    
    def _generate_education_level(self, age: int) -> str:
        """Generate education level with age-appropriate patterns."""
        if age < 18:
            if age < 6: return 'no_formal_education'
            elif age < 12: return 'primary_incomplete'
            elif age < 15: return 'primary_complete'
            else: return 'secondary_incomplete'
        else:
            weights = [0.01, 0.05, 0.15, 0.25, 0.30, 0.15, 0.09]  # Younger generation
            if age > 65: weights = [0.10, 0.25, 0.30, 0.20, 0.10, 0.04, 0.01] # Older generation
            elif age > 40: weights = [0.02, 0.10, 0.20, 0.30, 0.25, 0.10, 0.03] # Middle generation
            return np.random.choice(list(Config.EDUCATION_LEVELS.keys()), p=weights)

    def _generate_employment_status(self, age: int, education: str) -> str:
        """Enhanced employment status generation with education multipliers."""
        if age < 16: return 'student'
        if age >= 65: return 'retired'

        # Use the new employment multipliers
        multipliers = Config.EMPLOYMENT_MULTIPLIERS.get(education, {})
        base_probs = list(Config.EMPLOYMENT_STATUS.values())
        employment_statuses = list(Config.EMPLOYMENT_STATUS.keys())
        
        adjusted_probs = []
        for i, status in enumerate(employment_statuses):
            multiplier = multipliers.get(status, 1.0)
            adjusted_probs.append(base_probs[i] * multiplier)
        
        # Normalize probabilities
        total = sum(adjusted_probs)
        adjusted_probs = [p / total for p in adjusted_probs]
        
        return np.random.choice(employment_statuses, p=adjusted_probs)

    def _generate_
