import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date
import calendar
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Extending the base framework with industry-standard feature generators

class SociodemographicGenerator(BaseFeatureGenerator):
    """
    Industry-standard sociodemographic features commonly used in healthcare analytics
    Based on WHO recommendations and epidemiological best practices
    """
    
    def __init__(self):
        # Education levels based on UNESCO classification
        self.education_levels = {
            'no_formal_education': 0.05,
            'primary_incomplete': 0.15,
            'primary_complete': 0.20,
            'secondary_incomplete': 0.25,
            'secondary_complete': 0.20,
            'tertiary_incomplete': 0.10,
            'tertiary_complete': 0.05
        }
        
        # Employment status distribution
        self.employment_status = {
            'employed_full_time': 0.45,
            'employed_part_time': 0.15,
            'unemployed_seeking': 0.12,
            'unemployed_not_seeking': 0.08,
            'student': 0.10,
            'retired': 0.08,
            'disabled': 0.02
        }
        
        # Marital status distribution
        self.marital_status = {
            'single': 0.35,
            'married': 0.40,
            'divorced': 0.12,
            'separated': 0.05,
            'widowed': 0.08
        }
        
        # Urban/rural classification
        self.area_classification = {
            'urban': 0.70,
            'suburban': 0.20,
            'rural': 0.10
        }
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate comprehensive sociodemographic features"""
        age = person['age']
        sex = person['sex']
        
        # Education level (age-dependent)
        education = self._generate_education_level(age)
        
        # Employment status (age and education dependent)
        employment = self._generate_employment_status(age, education)
        
        # Marital status (age and sex dependent)
        marital = self._generate_marital_status(age, sex)
        
        # Household composition
        household_size = self._generate_household_size(age, marital)
        children_in_household = self._generate_children_count(age, marital)
        
        # Geographic and economic indicators
        area_type = np.random.choice(list(self.area_classification.keys()), 
                                   p=list(self.area_classification.values()))
        
        # Socioeconomic status proxy (composite of education, employment, area)
        ses_score = self._calculate_ses_score(education, employment, area_type)
        
        # Social support indicators
        social_support_score = self._generate_social_support_score(marital, household_size)
        
        return {
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
    
    def _generate_education_level(self, age: int) -> str:
        """Generate education level with age-appropriate patterns"""
        if age < 18:
            # Youth education patterns
            if age < 6:
                return 'no_formal_education'
            elif age < 12:
                return 'primary_incomplete'
            elif age < 15:
                return 'primary_complete'
            else:
                return 'secondary_incomplete'
        else:
            # Adult education distribution with generational effects
            if age > 65:
                # Older generation - lower education levels
                weights = [0.10, 0.25, 0.30, 0.20, 0.10, 0.04, 0.01]
            elif age > 40:
                # Middle generation
                weights = [0.02, 0.10, 0.20, 0.30, 0.25, 0.10, 0.03]
            else:
                # Younger generation - higher education levels
                weights = [0.01, 0.05, 0.15, 0.25, 0.30, 0.15, 0.09]
            
            levels = list(self.education_levels.keys())
            return np.random.choice(levels, p=weights)
    
    def _generate_employment_status(self, age: int, education: str) -> str:
        """Generate employment status based on age and education"""
        if age < 16:
            return 'student'
        elif age >= 65:
            return 'retired'
        else:
            # Education-adjusted employment probabilities
            education_multipliers = {
                'no_formal_education': {'employed_full_time': 0.6, 'unemployed_seeking': 1.8},
                'primary_incomplete': {'employed_full_time': 0.7, 'unemployed_seeking': 1.5},
                'primary_complete': {'employed_full_time': 0.8, 'unemployed_seeking': 1.3},
                'secondary_incomplete': {'employed_full_time': 0.9, 'unemployed_seeking': 1.1},
                'secondary_complete': {'employed_full_time': 1.0, 'unemployed_seeking': 1.0},
                'tertiary_incomplete': {'employed_full_time': 1.1, 'unemployed_seeking': 0.8},
                'tertiary_complete': {'employed_full_time': 1.3, 'unemployed_seeking': 0.5}
            }
            
            base_probs = list(self.employment_status.values())
            multipliers = education_multipliers.get(education, {})
            
            # Adjust probabilities based on education
            adjusted_probs = []
            for i, status in enumerate(self.employment_status.keys()):
                multiplier = multipliers.get(status, 1.0)
                adjusted_probs.append(base_probs[i] * multiplier)
            
            # Normalize
            total = sum(adjusted_probs)
            adjusted_probs = [p/total for p in adjusted_probs]
            
            return np.random.choice(list(self.employment_status.keys()), p=adjusted_probs)
    
    def _generate_marital_status(self, age: int, sex: str) -> str:
        """Generate marital status with age and sex patterns"""
        if age < 16:
            return 'single'
        
        # Age-adjusted probabilities
        if age < 25:
            probs = [0.80, 0.15, 0.03, 0.02, 0.00]  # Mostly single
        elif age < 35:
            probs = [0.45, 0.45, 0.05, 0.04, 0.01]  # Marriage peak
        elif age < 50:
            probs = [0.25, 0.55, 0.12, 0.06, 0.02]  # Stable marriage
        elif age < 65:
            probs = [0.20, 0.50, 0.15, 0.08, 0.07]  # Some widowhood
        else:
            probs = [0.15, 0.35, 0.10, 0.05, 0.35]  # Significant widowhood
        
        # Sex-specific adjustments (women live longer, different divorce patterns)
        if sex == 'F' and age > 65:
            probs[4] *= 1.5  # Higher widowhood for older women
            
        # Normalize
        total = sum(probs)
        probs = [p/total for p in probs]
        
        return np.random.choice(list(self.marital_status.keys()), p=probs)
    
    def _generate_household_size(self, age: int, marital: str) -> int:
        """Generate household size based on age and marital status"""
        if marital == 'single' and age > 25:
            return np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        elif marital in ['married', 'separated']:
            return np.random.choice([2, 3, 4, 5, 6], p=[0.3, 0.25, 0.25, 0.15, 0.05])
        elif marital == 'divorced':
            return np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        elif marital == 'widowed':
            return np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        else:
            return np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])
    
    def _generate_children_count(self, age: int, marital: str) -> int:
        """Generate number of children in household"""
        if age < 20 or marital == 'single':
            return np.random.choice([0, 1], p=[0.8, 0.2])
        elif marital in ['married', 'separated', 'divorced']:
            return np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        else:
            return np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    
    def _calculate_ses_score(self, education: str, employment: str, area_type: str) -> float:
        """Calculate socioeconomic status composite score"""
        # Education component (0-1)
        education_scores = {
            'no_formal_education': 0.0, 'primary_incomplete': 0.2, 'primary_complete': 0.3,
            'secondary_incomplete': 0.5, 'secondary_complete': 0.7, 
            'tertiary_incomplete': 0.8, 'tertiary_complete': 1.0
        }
        
        # Employment component (0-1)
        employment_scores = {
            'employed_full_time': 1.0, 'employed_part_time': 0.7, 'unemployed_seeking': 0.2,
            'unemployed_not_seeking': 0.1, 'student': 0.5, 'retired': 0.6, 'disabled': 0.3
        }
        
        # Area component (0-1)
        area_scores = {'urban': 0.8, 'suburban': 0.6, 'rural': 0.4}
        
        # Weighted composite
        education_weight = 0.5
        employment_weight = 0.3
        area_weight = 0.2
        
        ses_score = (education_scores[education] * education_weight + 
                    employment_scores[employment] * employment_weight + 
                    area_scores[area_type] * area_weight)
        
        return ses_score
    
    def _generate_social_support_score(self, marital: str, household_size: int) -> float:
        """Generate social support score based on social connections"""
        base_score = 0.5
        
        # Marital status adjustment
        marital_adjustments = {
            'married': 0.3, 'single': -0.1, 'divorced': -0.2,
            'separated': -0.2, 'widowed': -0.1
        }
        
        # Household size adjustment
        household_adjustment = min(0.2, (household_size - 1) * 0.1)
        
        support_score = base_score + marital_adjustments[marital] + household_adjustment
        
        # Add random variation
        support_score += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, support_score))
    
    def _is_primary_earner(self, employment: str, marital: str, sex: str) -> bool:
        """Determine if person is primary household earner"""
        if employment in ['unemployed_seeking', 'unemployed_not_seeking', 'disabled']:
            return False
        elif employment in ['employed_full_time']:
            return True
        elif marital == 'single':
            return employment in ['employed_full_time', 'employed_part_time']
        else:
            # In couples, model traditional and modern patterns
            if sex == 'M':
                return np.random.random() < 0.6  # 60% of men are primary earners
            else:
                return np.random.random() < 0.4  # 40% of women are primary earners
    
    def get_feature_names(self) -> List[str]:
        return [
            'education_level', 'employment_status', 'marital_status', 'household_size',
            'children_in_household', 'area_type', 'socioeconomic_status_score',
            'social_support_score', 'lives_alone', 'primary_earner'
        ]


class ClinicalComorbidityGenerator(BaseFeatureGenerator):
    """
    Clinical comorbidity features following ICD-11 and DSM-5 standards
    Common in electronic health records and clinical research
    """
    
    def __init__(self):
        # Mental health conditions prevalence (age and sex adjusted)
        self.mental_health_conditions = {
            'major_depression': 0.08,
            'anxiety_disorders': 0.12,
            'bipolar_disorder': 0.03,
            'personality_disorders': 0.05,
            'substance_use_disorder': 0.10,
            'psychotic_disorders': 0.01,
            'eating_disorders': 0.02,
            'ptsd': 0.04
        }
        
        # Physical health conditions
        self.physical_conditions = {
            'diabetes': 0.09,
            'hypertension': 0.25,
            'heart_disease': 0.06,
            'chronic_pain': 0.15,
            'cancer_history': 0.04,
            'neurological_disorder': 0.03,
            'chronic_kidney_disease': 0.02,
            'respiratory_disease': 0.08
        }
        
        # Medication classes (commonly prescribed)
        self.medication_classes = {
            'antidepressants': 0.15,
            'anxiolytics': 0.12,
            'antipsychotics': 0.03,
            'mood_stabilizers': 0.02,
            'stimulants': 0.04,
            'anticonvulsants': 0.05,
            'sedatives': 0.08
        }
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate clinical comorbidity features"""
        age = person['age']
        sex = person['sex']
        
        # Mental health conditions (higher rates in suicide attempt population)
        mh_conditions = self._generate_mental_health_conditions(age, sex)
        
        # Physical health conditions
        physical_conditions = self._generate_physical_conditions(age, sex)
        
        # Medication profile
        medications = self._generate_medication_profile(mh_conditions, physical_conditions)
        
        # Clinical severity indicators
        severity_indicators = self._generate_severity_indicators(mh_conditions, age)
        
        # Healthcare utilization
        utilization = self._generate_healthcare_utilization(mh_conditions, physical_conditions)
        
        # Combine all features
        features = {
            **mh_conditions,
            **physical_conditions,
            **medications,
            **severity_indicators,
            **utilization
        }
        
        # Add composite scores
        features.update(self._calculate_composite_scores(mh_conditions, physical_conditions))
        
        return features
    
    def _generate_mental_health_conditions(self, age: int, sex: str) -> Dict:
        """Generate mental health conditions with realistic prevalence"""
        conditions = {}
        
        for condition, base_rate in self.mental_health_conditions.items():
            # Age adjustments
            if condition == 'eating_disorders' and 15 <= age <= 25:
                adjusted_rate = base_rate * 3.0  # Peak in young adults
            elif condition == 'anxiety_disorders' and 25 <= age <= 45:
                adjusted_rate = base_rate * 1.5  # Higher in middle age
            elif condition == 'major_depression' and age > 65:
                adjusted_rate = base_rate * 1.3  # Higher in elderly
            else:
                adjusted_rate = base_rate
            
            # Sex adjustments
            if sex == 'F':
                if condition in ['major_depression', 'anxiety_disorders', 'eating_disorders']:
                    adjusted_rate *= 1.8  # Higher in females
                elif condition == 'substance_use_disorder':
                    adjusted_rate *= 0.6  # Lower in females
            else:  # Male
                if condition == 'substance_use_disorder':
                    adjusted_rate *= 1.5  # Higher in males
                elif condition in ['major_depression', 'anxiety_disorders']:
                    adjusted_rate *= 0.7  # Lower in males
            
            # Higher rates in suicide attempt population (selection bias)
            adjusted_rate *= 3.0  # Suicide attempt population has higher mental health burden
            adjusted_rate = min(0.8, adjusted_rate)  # Cap at 80%
            
            conditions[f'has_{condition}'] = np.random.random() < adjusted_rate
        
        return conditions
    
    def _generate_physical_conditions(self, age: int, sex: str) -> Dict:
        """Generate physical health conditions"""
        conditions = {}
        
        for condition, base_rate in self.physical_conditions.items():
            # Age adjustments
            if age < 30:
                age_multiplier = 0.3
            elif age < 50:
                age_multiplier = 0.8
            elif age < 65:
                age_multiplier = 1.5
            else:
                age_multiplier = 2.5
            
            # Condition-specific age patterns
            if condition == 'diabetes' and age < 40:
                age_multiplier *= 0.5
            elif condition == 'hypertension' and age < 35:
                age_multiplier *= 0.3
            elif condition == 'chronic_pain':
                age_multiplier = 1.0  # More constant across ages
            
            # Sex adjustments
            if sex == 'F' and condition == 'heart_disease':
                sex_multiplier = 0.7  # Lower in females (pre-menopause protection)
            elif sex == 'M' and condition == 'chronic_pain':
                sex_multiplier = 0.8  # Slightly lower in males
            else:
                sex_multiplier = 1.0
            
            adjusted_rate = base_rate * age_multiplier * sex_multiplier
            adjusted_rate = min(0.7, adjusted_rate)  # Cap at 70%
            
            conditions[f'has_{condition}'] = np.random.random() < adjusted_rate
        
        return conditions
    
    def _generate_medication_profile(self, mh_conditions: Dict, physical_conditions: Dict) -> Dict:
        """Generate medication profile based on conditions"""
        medications = {}
        
        for med_class, base_rate in self.medication_classes.items():
            # Condition-based prescribing patterns
            adjusted_rate = base_rate
            
            if med_class == 'antidepressants' and mh_conditions.get('has_major_depression'):
                adjusted_rate = 0.8  # 80% of depression patients on antidepressants
            elif med_class == 'anxiolytics' and mh_conditions.get('has_anxiety_disorders'):
                adjusted_rate = 0.6
            elif med_class == 'antipsychotics' and mh_conditions.get('has_psychotic_disorders'):
                adjusted_rate = 0.9
            elif med_class == 'mood_stabilizers' and mh_conditions.get('has_bipolar_disorder'):
                adjusted_rate = 0.7
            
            medications[f'prescribed_{med_class}'] = np.random.random() < adjusted_rate
        
        # Polypharmacy indicator
        total_medications = sum(medications.values())
        medications['polypharmacy'] = total_medications >= 3
        medications['medication_count'] = total_medications
        
        return medications
    
    def _generate_severity_indicators(self, mh_conditions: Dict, age: int) -> Dict:
        """Generate clinical severity indicators"""
        # GAF score (Global Assessment of Functioning) - 1-100 scale
        if any(mh_conditions.values()):
            # Lower functioning if mental health conditions present
            gaf_score = int(np.random.normal(55, 15))
        else:
            gaf_score = int(np.random.normal(75, 10))
        
        gaf_score = max(1, min(100, gaf_score))
        
        # Clinical severity based on number of conditions
        condition_count = sum(mh_conditions.values())
        if condition_count >= 3:
            severity = 'severe'
        elif condition_count >= 2:
            severity = 'moderate'
        elif condition_count >= 1:
            severity = 'mild'
        else:
            severity = 'minimal'
        
        # Functional impairment indicators
        functional_impairment = gaf_score < 60  # Moderate to severe impairment
        
        return {
            'gaf_score': gaf_score,
            'clinical_severity': severity,
            'functional_impairment': functional_impairment,
            'comorbidity_count_mental': sum(mh_conditions.values()),
            'complex_case': condition_count >= 2 and functional_impairment
        }
    
    def _generate_healthcare_utilization(self, mh_conditions: Dict, physical_conditions: Dict) -> Dict:
        """Generate healthcare utilization patterns"""
        # Base utilization rates
        total_conditions = sum(mh_conditions.values()) + sum(physical_conditions.values())
        
        # ED visits in past year (Poisson distribution)
        ed_visits_lambda = 0.5 + total_conditions * 0.3
        ed_visits_past_year = np.random.poisson(ed_visits_lambda)
        
        # Hospitalizations in past year
        hosp_lambda = 0.1 + total_conditions * 0.15
        hospitalizations_past_year = np.random.poisson(hosp_lambda)
        
        # Outpatient visits
        outpatient_lambda = 3 + total_conditions * 2
        outpatient_visits_past_year = np.random.poisson(outpatient_lambda)
        
        # High utilizer indicator
        high_utilizer = (ed_visits_past_year >= 4 or 
                        hospitalizations_past_year >= 2 or 
                        outpatient_visits_past_year >= 15)
        
        return {
            'ed_visits_past_year': ed_visits_past_year,
            'hospitalizations_past_year': hospitalizations_past_year,
            'outpatient_visits_past_year': outpatient_visits_past_year,
            'high_healthcare_utilizer': high_utilizer,
            'total_healthcare_encounters': (ed_visits_past_year + 
                                          hospitalizations_past_year + 
                                          outpatient_visits_past_year)
        }
    
    def _calculate_composite_scores(self, mh_conditions: Dict, physical_conditions: Dict) -> Dict:
        """Calculate composite clinical scores"""
        # Charlson Comorbidity Index (simplified)
        charlson_score = sum(physical_conditions.values())  # Simplified version
        
        # Mental health burden score
        mh_burden_score = sum(mh_conditions.values())
        
        # Overall clinical complexity
        total_burden = charlson_score + mh_burden_score
        
        if total_burden >= 5:
            complexity = 'high'
        elif total_burden >= 3:
            complexity = 'moderate'
        elif total_burden >= 1:
            complexity = 'low'
        else:
            complexity = 'minimal'
        
        return {
            'charlson_comorbidity_index': charlson_score,
            'mental_health_burden_score': mh_burden_score,
            'clinical_complexity': complexity,
            'total_comorbidity_burden': total_burden
        }
    
    def get_feature_names(self) -> List[str]:
        # Mental health conditions
        mh_features = [f'has_{condition}' for condition in self.mental_health_conditions.keys()]
        
        # Physical conditions
        physical_features = [f'has_{condition}' for condition in self.physical_conditions.keys()]
        
        # Medications
        med_features = [f'prescribed_{med}' for med in self.medication_classes.keys()]
        med_features.extend(['polypharmacy', 'medication_count'])
        
        # Severity and utilization
        other_features = [
            'gaf_score', 'clinical_severity', 'functional_impairment', 
            'comorbidity_count_mental', 'complex_case',
            'ed_visits_past_year', 'hospitalizations_past_year', 
            'outpatient_visits_past_year', 'high_healthcare_utilizer',
            'total_healthcare_encounters', 'charlson_comorbidity_index',
            'mental_health_burden_score', 'clinical_complexity', 
            'total_comorbidity_burden'
        ]
        
        return mh_features + physical_features + med_features + other_features


class RiskFactorGenerator(BaseFeatureGenerator):
    """
    Evidence-based risk factor features from suicide prevention literature
    Based on WHO, CDC, and clinical practice guidelines
    """
    
    def __init__(self):
        # Risk factor categories with evidence weights
        self.risk_factors = {
            'hopelessness_scale_score': {'mean': 12, 'std': 6, 'range': (0, 20)},
            'depression_severity_phq9': {'mean': 15, 'std': 5, 'range': (0, 27)},
            'anxiety_severity_gad7': {'mean': 12, 'std': 4, 'range': (0, 21)},
            'suicidal_ideation_intensity': {'mean': 6, 'std': 3, 'range': (0, 10)},
            'social_connectedness_score': {'mean': 4, 'std': 2, 'range': (0, 10)},
            'life_stress_score': {'mean': 7, 'std': 2, 'range': (0, 10)}
        }
        
        # Protective factors
        self.protective_factors = {
            'reasons_for_living_score': {'mean': 35, 'std': 8, 'range': (0, 48)},
            'coping_skills_score': {'mean': 25, 'std': 6, 'range': (0, 40)},
            'treatment_alliance_score': {'mean': 7, 'std': 2, 'range': (0, 10)}
        }
        
        # Binary risk indicators
        self.binary_risk_indicators = {
            'recent_loss_bereavement': 0.15,
            'relationship_breakup_recent': 0.20,
            'job_loss_recent': 0.12,
            'legal_problems': 0.08,
            'financial_crisis': 0.25,
            'academic_failure': 0.10,
            'social_isolation': 0.30,
            'bullying_victim': 0.08,
            'domestic_violence_exposure': 0.12,
            'childhood_trauma_history': 0.35
        }
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate evidence-based risk and protective factors"""
        age = person['age']
        sex = person['sex']
        
        # Generate psychological risk factors
        risk_scores = self._generate_psychological_scores(age, sex)
        
        # Generate protective factors
        protective_scores = self._generate_protective_factors(age, sex)
        
        # Generate life event stressors
        life_events = self._generate_life_events(age, sex)
        
        # Generate trauma history
        trauma_history = self._generate_trauma_history(age, sex)
        
        # Calculate composite risk indices
        composite_indices = self._calculate_risk_indices(risk_scores, protective_scores, life_events)
        
        # Combine all features
        features = {
            **risk_scores,
            **protective_scores,
            **life_events,
            **trauma_history,
            **composite_indices
        }
        
        return features
    
    def _generate_psychological_scores(self, age: int, sex: str) -> Dict:
        """Generate validated psychological assessment scores"""
        scores = {}
        
        for factor, params in self.risk_factors.items():
            mean = params['mean']
            std = params['std']
            min_val, max_val = params['range']
            
            # Age adjustments
            if factor == 'hopelessness_scale_score' and age < 25:
                mean += 2  # Higher hopelessness in youth
            elif factor == 'depression_severity_phq9' and age > 65:
                mean += 1.5  # Higher depression in elderly
            elif factor == 'anxiety_severity_gad7' and 25 <= age <= 45:
                mean += 1  # Peak anxiety in middle age
            
            # Sex adjustments
            if sex == 'F':
                if factor in ['depression_severity_phq9', 'anxiety_severity_gad7']:
                    mean += 1.5  # Higher in females
            else:  # Male
                if factor == 'hopelessness_scale_score':
                    mean += 1  # Slightly higher hopelessness in males
            
            # Generate score with bounds
            score = np.random.normal(mean, std)
            score = max(min_val, min(max_val, score))
            scores[factor] = round(score, 1)
        
        return scores
    
    def _generate_protective_factors(self, age: int, sex: str) -> Dict:
        """Generate protective factor scores"""
        scores = {}
        
        for factor, params in self.protective_factors.items():
            mean = params['mean']
            std = params['std']
            min_val, max_val = params['range']
            
            # Age adjustments
            if factor == 'reasons_for_living_score':
                if age < 20:
                    mean -= 3  # Lower reasons for living in youth
                elif age > 60:
                    mean += 2  # Higher in older adults
            elif factor == 'coping_skills_score' and age > 40:
                mean += 2  # Better coping with age/experience
            
            # Sex adjustments
            if sex == 'F' and factor == 'social_connectedness_score':
                mean += 1  # Females often have stronger social connections
            
            # Generate score with bounds
            score = np.random.normal(mean, std)
            score = max(min_val, min(max_val, score))
            scores[factor] = round(score, 1)
        
        return scores
    
    def _generate_life_events(self, age: int, sex: str) -> Dict:
        """Generate recent life events and stressors"""
        events = {}
        
        for event, base_rate in self.binary_risk_indicators.items():
            # Age-specific adjustments
            adjusted_rate = base_rate
            
            if event == 'relationship_breakup_recent' and 16 <= age <= 35:
                adjusted_rate *= 2.0  # Higher in young adults
            elif event == 'job_loss_recent' and 25 <= age <= 55:
                adjusted_rate *= 1.5  # Higher in working age
            elif event == 'academic_failure' and 15 <= age <= 25:
                adjusted_rate *= 3.0  # Higher in students
            elif event == 'financial_crisis' and 30 <= age <= 60:
                adjusted_rate *= 1.3  # Higher in prime working years
            elif event == 'social_isolation' and age > 65:
                adjusted_rate *= 1.8  # Higher in elderly
            
            # Sex-specific adjustments
            if sex == 'F':
                if event in ['domestic_violence_exposure', 'bullying_victim']:
                    adjusted_rate *= 1.4  # Higher reported rates in females
            else:  # Male
                if event in ['job_loss_recent', 'legal_problems']:
                    adjusted_rate *= 1.2  # Slightly higher in males
            
            # Higher rates in suicide attempt population
            adjusted_rate *= 2.5  # Selection bias - higher stress in attempt population
            adjusted_rate = min(0.8, adjusted_rate)  # Cap at 80%
            
            events[event] = np.random.random() < adjusted_rate
        
        return events
    
    def _generate_trauma_history(self, age: int, sex: str) -> Dict:
        """Generate trauma and adverse childhood experiences"""
        trauma = {}
        
        # ACE (Adverse Childhood Experiences) categories
        ace_categories = {
            'physical_abuse_childhood': 0.28,
            'emotional_abuse_childhood': 0.35,
            'sexual_abuse_childhood': 0.20,
            'physical_neglect_childhood': 0.16,
            'emotional_neglect_childhood': 0.18,
            'household_dysfunction_childhood': 0.25,
            'parental_substance_abuse': 0.30,
            'parental_mental_illness': 0.20,
            'domestic_violence_witnessed': 0.12
        }
        
        ace_score = 0
        for ace_type, base_rate in ace_categories.items():
            # Sex-specific adjustments
            if sex == 'F' and ace_type == 'sexual_abuse_childhood':
                adjusted_rate = base_rate * 2.5  # Higher rates in females
            elif sex == 'M' and ace_type == 'physical_abuse_childhood':
                adjusted_rate = base_rate * 1.3  # Slightly higher in males
            else:
                adjusted_rate = base_rate
            
            # Higher rates in suicide attempt population
            adjusted_rate *= 2.0  # ACEs more common in suicide attempt population
            adjusted_rate = min(0.7, adjusted_rate)  # Cap at 70%
            
            has_ace = np.random.random() < adjusted_rate
            trauma[ace_type] = has_ace
            if has_ace:
                ace_score += 1
        
        trauma['ace_score'] = ace_score
        trauma['high_ace_score'] = ace_score >= 4  # Clinical threshold
        
        # Adult trauma exposure
        adult_trauma_types = {
            'combat_exposure': 0.05,
            'serious_accident': 0.15,
            'natural_disaster': 0.08,
            'violent_crime_victim': 0.12,
            'sexual_assault_adult': 0.10
        }
        
        adult_trauma_count = 0
        for trauma_type, base_rate in adult_trauma_types.items():
            # Age adjustments
            if trauma_type == 'combat_exposure' and sex == 'M':
                adjusted_rate = base_rate * 3.0  # Higher in males
            elif trauma_type == 'sexual_assault_adult' and sex == 'F':
                adjusted_rate = base_rate * 4.0  # Much higher in females
            else:
                adjusted_rate = base_rate
            
            # Higher in suicide attempt population
            adjusted_rate *= 1.8
            adjusted_rate = min(0.6, adjusted_rate)
            
            has_trauma = np.random.random() < adjusted_rate
            trauma[trauma_type] = has_trauma
            if has_trauma:
                adult_trauma_count += 1
        
        trauma['adult_trauma_count'] = adult_trauma_count
        trauma['polytraumatized'] = (ace_score >= 2 and adult_trauma_count >= 1)
        
        return trauma
    
    def _calculate_risk_indices(self, risk_scores: Dict, protective_scores: Dict, 
                              life_events: Dict) -> Dict:
        """Calculate composite risk indices"""
        # Weighted risk index (0-100 scale)
        risk_components = [
            risk_scores['hopelessness_scale_score'] * 2.5,  # Weight: 2.5
            risk_scores['depression_severity_phq9'] * 1.8,  # Weight: 1.8
            risk_scores['suicidal_ideation_intensity'] * 5.0,  # Weight: 5.0 (highest)
            (10 - risk_scores['social_connectedness_score']) * 2.0,  # Inverted
            risk_scores['life_stress_score'] * 2.0
        ]
        
        weighted_risk_index = sum(risk_components)
        weighted_risk_index = min(100, max(0, weighted_risk_index))
        
        # Protective index (0-100 scale)
        protective_components = [
            protective_scores['reasons_for_living_score'] * 1.5,
            protective_scores['coping_skills_score'] * 1.8,
            protective_scores['treatment_alliance_score'] * 4.0
        ]
        
        protective_index = sum(protective_components)
        protective_index = min(100, max(0, protective_index))
        
        # Net risk (risk - protective factors)
        net_risk_score = weighted_risk_index - (protective_index * 0.7)
        net_risk_score = max(0, net_risk_score)
        
        # Acute risk indicators
        acute_stressors = sum(life_events.values())
        acute_risk_elevation = acute_stressors >= 3
        
        # Risk categorization
        if net_risk_score >= 70 or acute_risk_elevation:
            risk_category = 'high'
        elif net_risk_score >= 45:
            risk_category = 'moderate'
        elif net_risk_score >= 20:
            risk_category = 'low'
        else:
            risk_category = 'minimal'
        
        return {
            'weighted_risk_index': round(weighted_risk_index, 1),
            'protective_factor_index': round(protective_index, 1),
            'net_risk_score': round(net_risk_score, 1),
            'acute_stressor_count': acute_stressors,
            'acute_risk_elevation': acute_risk_elevation,
            'overall_risk_category': risk_category,
            'risk_protective_ratio': round(weighted_risk_index / max(1, protective_index), 2)
        }
    
    def get_feature_names(self) -> List[str]:
        risk_features = list(self.risk_factors.keys())
        protective_features = list(self.protective_factors.keys())
        event_features = list(self.binary_risk_indicators.keys())
        
        trauma_features = [
            'physical_abuse_childhood', 'emotional_abuse_childhood', 'sexual_abuse_childhood',
            'physical_neglect_childhood', 'emotional_neglect_childhood', 'household_dysfunction_childhood',
            'parental_substance_abuse', 'parental_mental_illness', 'domestic_violence_witnessed',
            'ace_score', 'high_ace_score', 'combat_exposure', 'serious_accident',
            'natural_disaster', 'violent_crime_victim', 'sexual_assault_adult',
            'adult_trauma_count', 'polytraumatized'
        ]
        
        composite_features = [
            'weighted_risk_index', 'protective_factor_index', 'net_risk_score',
            'acute_stressor_count', 'acute_risk_elevation', 'overall_risk_category',
            'risk_protective_ratio'
        ]
        
        return risk_features + protective_features + event_features + trauma_features + composite_features


class HealthcareSystemGenerator(BaseFeatureGenerator):
    """
    Healthcare system performance and quality indicators
    Common in health services research and quality improvement
    """
    
    def __init__(self):
        # Healthcare quality metrics
        self.quality_indicators = {
            'care_coordination_score': {'mean': 7.5, 'std': 2.0, 'range': (0, 10)},
            'treatment_accessibility_score': {'mean': 6.8, 'std': 2.5, 'range': (0, 10)},
            'provider_communication_score': {'mean': 8.0, 'std': 1.5, 'range': (0, 10)},
            'continuity_of_care_score': {'mean': 7.2, 'std': 2.2, 'range': (0, 10)}
        }
        
        # System performance metrics
        self.performance_metrics = {
            'wait_time_initial_appointment_days': {'lambda': 14},  # Poisson
            'treatment_delay_days': {'lambda': 7},
            'missed_appointments_count': {'lambda': 2},
            'provider_changes_count': {'lambda': 1.5}
        }
        
        # Care setting types
        self.care_settings = {
            'primary_care': 0.85,
            'community_mental_health': 0.60,
            'specialist_psychiatric': 0.35,
            'emergency_services': 0.25,
            'inpatient_psychiatric': 0.15,
            'intensive_outpatient': 0.20,
            'peer_support_services': 0.30
        }
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate healthcare system performance features"""
        
        # Quality indicators
        quality_scores = self._generate_quality_indicators(person)
        
        # Performance metrics
        performance_metrics = self._generate_performance_metrics(person)
        
        # Care settings and access
        care_access = self._generate_care_access_features(person, attempt_date)
        
        # System integration features
        integration_features = self._generate_integration_features(person)
        
        # Insurance and coverage features
        coverage_features = self._generate_coverage_features(person)
        
        return {
            **quality_scores,
            **performance_metrics,
            **care_access,
            **integration_features,
            **coverage_features
        }
    
    def _generate_quality_indicators(self, person: Dict) -> Dict:
        """Generate healthcare quality indicator scores"""
        scores = {}
        
        for indicator, params in self.quality_indicators.items():
            mean = params['mean']
            std = params['std']
            min_val, max_val = params['range']
            
            # Adjust based on healthcare provider type
            if person['healthcare_provider'] == 'private':
                mean += 0.5  # Slightly higher quality scores in private care
            
            # Adjust based on area type
            if 'area_type' in person and person['area_type'] == 'rural':
                mean -= 1.0  # Lower scores in rural areas
            elif 'area_type' in person and person['area_type'] == 'urban':
                mean += 0.3  # Slightly higher in urban areas
            
            score = np.random.normal(mean, std)
            score = max(min_val, min(max_val, score))
            scores[indicator] = round(score, 1)
        
        return scores
    
    def _generate_performance_metrics(self, person: Dict) -> Dict:
        """Generate system performance metrics"""
        metrics = {}
        
        for metric, params in self.performance_metrics.items():
            lambda_param = params['lambda']
            
            # Adjust lambda based on system characteristics
            if person['healthcare_provider'] == 'public':
                lambda_param *= 1.3  # Longer waits in public system
            
            if 'area_type' in person and person['area_type'] == 'rural':
                lambda_param *= 1.5  # Longer waits in rural areas
            
            value = np.random.poisson(lambda_param)
            metrics[metric] = value
        
        # High utilizer adjustments
        if 'high_healthcare_utilizer' in person and person['high_healthcare_utilizer']:
            metrics['missed_appointments_count'] *= 2
            metrics['provider_changes_count'] += 1
        
        return metrics
    
    def _generate_care_access_features(self, person: Dict, attempt_date: date) -> Dict:
        """Generate care access and utilization features"""
        features = {}
        
        # Care settings accessed
        for setting, base_prob in self.care_settings.items():
            # Adjust probabilities based on person characteristics
            adjusted_prob = base_prob
            
            if person['healthcare_provider'] == 'private':
                if setting in ['specialist_psychiatric', 'intensive_outpatient']:
                    adjusted_prob *= 1.4  # Better access to specialized care
            
            if setting == 'emergency_services':
                adjusted_prob *= 2.0  # Higher in suicide attempt population
            
            features[f'accessed_{setting}'] = np.random.random() < adjusted_prob
        
        # Geographic access barriers
        features['travel_distance_to_care_km'] = self._generate_travel_distance(person)
        features['transportation_barriers'] = np.random.random() < 0.25
        
        # Appointment scheduling
        features['same_day_access_available'] = np.random.random() < 0.30
        features['after_hours_access_available'] = np.random.random() < 0.40
        features['telehealth_available'] = np.random.random() < 0.70
        
        # Language and cultural barriers
        features['language_barrier_present'] = (person['nationality'] == 'Foreign' and 
                                              np.random.random() < 0.60)
        features['cultural_competency_rating'] = round(np.random.normal(7.0, 1.5), 1)
        features['cultural_competency_rating'] = max(0, min(10, features['cultural_competency_rating']))
        
        return features
    
    def _generate_travel_distance(self, person: Dict) -> float:
        """Generate travel distance to mental health care"""
        if 'area_type' in person:
            if person['area_type'] == 'urban':
                distance = np.random.gamma(2, 3)  # Mean ~6km
            elif person['area_type'] == 'suburban':
                distance = np.random.gamma(3, 5)  # Mean ~15km
            else:  # rural
                distance = np.random.gamma(4, 10)  # Mean ~40km
        else:
            distance = np.random.gamma(3, 5)  # Default
        
        return round(distance, 1)
    
    def _generate_integration_features(self, person: Dict) -> Dict:
        """Generate care integration and coordination features"""
        features = {}
        
        # Electronic health record integration
        features['ehr_integrated'] = np.random.random() < 0.75
        features['care_plan_documented'] = np.random.random() < 0.65
        features['crisis_plan_available'] = np.random.random() < 0.45
        
        # Team-based care
        features['multidisciplinary_team'] = np.random.random() < 0.55
        features['case_manager_assigned'] = np.random.random() < 0.40
        features['peer_support_available'] = np.random.random() < 0.35
        
        # Information sharing
        features['provider_communication_quality'] = round(np.random.normal(6.5, 2.0), 1)
        features['provider_communication_quality'] = max(0, min(10, features['provider_communication_quality']))
        
        # Care transitions
        features['discharge_planning_quality'] = round(np.random.normal(7.0, 1.8), 1)
        features['discharge_planning_quality'] = max(0, min(10, features['discharge_planning_quality']))
        
        return features
    
    def _generate_coverage_features(self, person: Dict) -> Dict:
        """Generate insurance coverage and financial features"""
        features = {}
        
        # Insurance type (Uruguay context)
        if person['healthcare_provider'] == 'private':
            insurance_types = ['private_insurance', 'employer_coverage', 'supplemental']
            features['primary_insurance'] = np.random.choice(insurance_types)
        else:
            features['primary_insurance'] = 'public_snis'
        
        # Coverage adequacy
        features['mental_health_coverage_adequate'] = np.random.random() < 0.70
        features['medication_coverage_adequate'] = np.random.random() < 0.80
        features['therapy_sessions_covered'] = np.random.randint(8, 25)  # Sessions per year
        
        # Financial barriers
        if 'socioeconomic_status_score' in person:
            ses_score = person['socioeconomic_status_score']
            financial_barrier_prob = 0.5 - (ses_score * 0.4)  # Lower SES = higher barriers
        else:
            financial_barrier_prob = 0.30
        
        features['financial_barriers_to_care'] = np.random.random() < financial_barrier_prob
        features['medication_cost_burden'] = np.random.random() < (financial_barrier_prob * 0.8)
        
        # Cost-related delays
        features['delayed_care_due_to_cost'] = features['financial_barriers_to_care'] and np.random.random() < 0.60
        features['skipped_medications_due_to_cost'] = features['medication_cost_burden'] and np.random.random() < 0.40
        
        return features
    
    def get_feature_names(self) -> List[str]:
        quality_features = list(self.quality_indicators.keys())
        performance_features = list(self.performance_metrics.keys())
        
        access_features = [
            f'accessed_{setting}' for setting in self.care_settings.keys()
        ] + [
            'travel_distance_to_care_km', 'transportation_barriers',
            'same_day_access_available', 'after_hours_access_available',
            'telehealth_available', 'language_barrier_present',
            'cultural_competency_rating'
        ]
        
        integration_features = [
            'ehr_integrated', 'care_plan_documented', 'crisis_plan_available',
            'multidisciplinary_team', 'case_manager_assigned', 'peer_support_available',
            'provider_communication_quality', 'discharge_planning_quality'
        ]
        
        coverage_features = [
            'primary_insurance', 'mental_health_coverage_adequate',
            'medication_coverage_adequate', 'therapy_sessions_covered',
            'financial_barriers_to_care', 'medication_cost_burden',
            'delayed_care_due_to_cost', 'skipped_medications_due_to_cost'
        ]
        
        return quality_features + performance_features + access_features + integration_features + coverage_features


class MachineLearningGenerator(BaseFeatureGenerator):
    """
    Machine learning and model development features
    Industry standard features for ML pipelines in healthcare
    """
    
    def __init__(self):
        # Feature engineering configurations
        self.encoding_schemes = ['ordinal', 'one_hot', 'target', 'frequency']
        self.scaling_methods = ['standard', 'minmax', 'robust', 'quantile']
        self.imputation_strategies = ['mean', 'median', 'mode', 'knn', 'iterative']
        
        # Cross-validation configurations
        self.cv_strategies = ['stratified_kfold', 'time_series_split', 'group_kfold']
        self.n_folds_options = [3, 5, 10]
        
        # Model selection parameters
        self.model_families = [
            'logistic_regression', 'random_forest', 'gradient_boosting',
            'neural_network', 'svm', 'naive_bayes', 'ensemble'
        ]
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate ML-specific features for model development"""
        
        # Data splitting features
        split_features = self._generate_data_splits(person, attempt_date)
        
        # Feature engineering metadata
        engineering_features = self._generate_feature_engineering_metadata(person)
        
        # Model training features
        training_features = self._generate_training_metadata(attempt_date)
        
        # Prediction and scoring features
        prediction_features = self._generate_prediction_features(person)
        
        # Model interpretability features
        interpretability_features = self._generate_interpretability_features(person)
        
        return {
            **split_features,
            **engineering_features,
            **training_features,
            **prediction_features,
            **interpretability_features
        }
    
    def _generate_data_splits(self, person: Dict, attempt_date: date) -> Dict:
        """Generate train/validation/test splits with multiple strategies"""
        features = {}
        
        # Temporal splits
        year = attempt_date.year
        if year <= 2024:
            temporal_split = 'train'
        elif year == 2025:
            temporal_split = 'validation'
        elif year == 2026:
            temporal_split = 'test'
        else:
            temporal_split = 'holdout'
        
        # Random splits (consistent based on person_id)
        person_hash = int(hashlib.md5(person['person_id'].encode()).hexdigest(), 16)
        random_seed = person_hash % 1000
        np.random.seed(random_seed)
        
        random_split = np.random.choice(['train', 'validation', 'test'], p=[0.7, 0.15, 0.15])
        
        # Stratified splits (by risk category if available)
        if 'overall_risk_category' in person:
            risk_category = person['overall_risk_category']
            if risk_category == 'high':
                stratified_split = np.random.choice(['train', 'validation', 'test'], p=[0.6, 0.2, 0.2])
            else:
                stratified_split = np.random.choice(['train', 'validation', 'test'], p=[0.75, 0.125, 0.125])
        else:
            stratified_split = random_split
        
        # Geographic splits
        if 'geographic_cluster_id' in person:
            cluster_hash = hash(person['geographic_cluster_id']) % 10
            if cluster_hash < 7:
                geographic_split = 'train'
            elif cluster_hash < 9:
                geographic_split = 'validation'
            else:
                geographic_split = 'test'
        else:
            geographic_split = random_split
        
        # Cross-validation fold assignment
        cv_fold = person_hash % 5  # 5-fold CV
        
        features.update({
            'temporal_split': temporal_split,
            'random_split': random_split,
            'stratified_split': stratified_split,
            'geographic_split': geographic_split,
            'cv_fold': cv_fold,
            'is_train_set': temporal_split == 'train',
            'is_validation_set': temporal_split == 'validation',
            'is_test_set': temporal_split == 'test'
        })
        
        return features
    
    def _generate_feature_engineering_metadata(self, person: Dict) -> Dict:
        """Generate feature engineering and preprocessing metadata"""
        features = {}
        
        # Encoding strategies for categorical variables
        features['categorical_encoding_strategy'] = np.random.choice(self.encoding_schemes)
        features['numerical_scaling_method'] = np.random.choice(self.scaling_methods)
        features['missing_value_strategy'] = np.random.choice(self.imputation_strategies)
        
        # Feature selection indicators
        features['high_cardinality_features'] = np.random.randint(0, 5)  # Number of high-cardinality features
        features['sparse_features_count'] = np.random.randint(0, 10)  # Number of sparse features
        features['engineered_features_count'] = np.random.randint(5, 25)  # Number of engineered features
        
        # Data quality flags
        features['data_quality_flag'] = np.random.choice(['clean', 'minor_issues', 'major_issues'], 
                                                        p=[0.7, 0.25, 0.05])
        features['outlier_score'] = round(np.random.beta(2, 8), 3)  # 0-1, lower is more normal
        features['missing_data_percentage'] = round(np.random.beta(1, 9) * 20, 1)  # 0-20%
        
        # Feature importance preprocessing
        features['feature_correlation_max'] = round(np.random.beta(3, 5), 3)  # Max correlation with other features
        features['target_correlation'] = round(np.random.normal(0, 0.3), 3)  # Correlation with target
        features['variance_inflation_factor'] = round(np.random.gamma(2, 2), 2)  # VIF for multicollinearity
        
        return features
    
    def _generate_training_metadata(self, attempt_date: date) -> Dict:
        """Generate model training metadata"""
        features = {}
        
        # Model selection
        features['primary_model_family'] = np.random.choice(self.model_families)
        features['ensemble_method'] = np.random.choice(['voting', 'stacking', 'blending', 'none'], 
                                                      p=[0.25, 0.25, 0.15, 0.35])
        
        # Cross-validation configuration
        features['cv_strategy'] = np.random.choice(self.cv_strategies)
        features['cv_folds'] = np.random.choice(self.n_folds_options)
        
        # Hyperparameter optimization
        features['hyperparameter_tuning_method'] = np.random.choice([
            'grid_search', 'random_search', 'bayesian_optimization', 'none'
        ], p=[0.2, 0.3, 0.3, 0.2])
        
        features['optimization_metric'] = np.random.choice([
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr'
        ])
        
        # Training characteristics
        features['training_time_minutes'] = np.random.gamma(3, 10)  # Training time
        features['convergence_achieved'] = np.random.random() < 0.85  # Model converged
        features['early_stopping_triggered'] = np.random.random() < 0.30  # Early stopping used
        
        return features
    
    def _generate_prediction_features(self, person: Dict) -> Dict:
        """Generate prediction and scoring features"""
        features = {}
        
        # Prediction scores (multiple models/horizons)
        prediction_horizons = ['7_day', '30_day', '90_day', '6_month', '1_year']
        
        for horizon in prediction_horizons:
            # Base probability adjusted by risk factors
            base_prob = 0.15  # Base 15% risk
            
            # Adjust based on known risk factors
            if 'overall_risk_category' in person:
                if person['overall_risk_category'] == 'high':
                    base_prob = 0.45
                elif person['overall_risk_category'] == 'moderate':
                    base_prob = 0.25
                elif person['overall_risk_category'] == 'low':
                    base_prob = 0.10
            
            # Time decay for longer horizons
            time_decay = {'7_day': 1.0, '30_day': 0.8, '90_day': 0.6, '6_month': 0.4, '1_year': 0.3}
            adjusted_prob = base_prob * time_decay[horizon]
            
            # Add model uncertainty
            prediction_score = np.random.beta(adjusted_prob * 10, (1 - adjusted_prob) * 10)
            features[f'prediction_score_{horizon}'] = round(prediction_score, 4)
            
            # Confidence intervals
            uncertainty = np.random.beta(2, 8) * 0.2  # 0-0.2 uncertainty
            features[f'prediction_lower_ci_{horizon}'] = round(max(0, prediction_score - uncertainty), 4)
            features[f'prediction_upper_ci_{horizon}'] = round(min(1, prediction_score + uncertainty), 4)
        
        # Model agreement and ensemble features
        features['model_consensus_score'] = round(np.random.beta(4, 2), 3)  # Agreement between models
        features['prediction_stability'] = round(np.random.beta(3, 2), 3)  # Stability across time
        features['ensemble_weight'] = round(np.random.beta(5, 2), 3)  # Weight in ensemble model
        
        # Calibration features
        features['calibration_score'] = round(np.random.beta(4, 3), 3)  # How well-calibrated predictions are
        features['over_confident'] = features['model_consensus_score'] > 0.9 and features['calibration_score'] < 0.7
        features['under_confident'] = features['model_consensus_score'] < 0.5 and features['calibration_score'] > 0.8
        
        return features
    
    def _generate_interpretability_features(self, person: Dict) -> Dict:
        """Generate model interpretability and explainability features"""
        features = {}
        
        # SHAP-like feature importance values
        important_features = [
            'age_importance', 'sex_importance', 'previous_attempts_importance',
            'mental_health_treatment_importance', 'depression_severity_importance',
            'social_support_importance', 'hopelessness_importance'
        ]
        
        # Generate importance scores that sum to 1
        importance_values = np.random.dirichlet([2, 2, 4, 3, 4, 3, 4])  # Higher weight on clinical factors
        
        for i, feature_name in enumerate(important_features):
            features[feature_name] = round(importance_values[i], 4)
        
        # Local explanations
        features['local_explanation_confidence'] = round(np.random.beta(3, 2), 3)
        features['counterfactual_distance'] = round(np.random.gamma(2, 0.5), 3)  # How far to change prediction
        
        # Global model insights
        features['model_complexity_score'] = round(np.random.beta(3, 3), 3)  # 0=simple, 1=complex
        features['linear_separability'] = round(np.random.beta(2, 3), 3)  # How linearly separable the data is
        features['feature_interaction_strength'] = round(np.random.beta(2, 4), 3)  # Strength of feature interactions
        
        # Clinical decision support features
        features['clinical_rule_triggered'] = np.random.choice([
            'high_risk_youth', 'multiple_attempts_history', 'acute_stressor_present',
            'treatment_non_adherence', 'none'
        ], p=[0.2, 0.25, 0.3, 0.15, 0.1])
        
        features['recommendation_type'] = np.random.choice([
            'immediate_intervention', 'enhanced_monitoring', 'standard_follow_up',
            'referral_specialist', 'crisis_team_activation'
        ], p=[0.15, 0.25, 0.35, 0.2, 0.05])
        
        # Uncertainty quantification
        features['epistemic_uncertainty'] = round(np.random.beta(2, 5), 3)  # Model uncertainty
        features['aleatoric_uncertainty'] = round(np.random.beta(3, 4), 3)  # Data uncertainty
        features['total_uncertainty'] = round(features['epistemic_uncertainty'] + features['aleatoric_uncertainty'], 3)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        split_features = [
            'temporal_split', 'random_split', 'stratified_split', 'geographic_split',
            'cv_fold', 'is_train_set', 'is_validation_set', 'is_test_set'
        ]
        
        engineering_features = [
            'categorical_encoding_strategy', 'numerical_scaling_method', 'missing_value_strategy',
            'high_cardinality_features', 'sparse_features_count', 'engineered_features_count',
            'data_quality_flag', 'outlier_score', 'missing_data_percentage',
            'feature_correlation_max', 'target_correlation', 'variance_inflation_factor'
        ]
        
        training_features = [
            'primary_model_family', 'ensemble_method', 'cv_strategy', 'cv_folds',
            'hyperparameter_tuning_method', 'optimization_metric', 'training_time_minutes',
            'convergence_achieved', 'early_stopping_triggered'
        ]
        
        prediction_features = []
        horizons = ['7_day', '30_day', '90_day', '6_month', '1_year']
        for horizon in horizons:
            prediction_features.extend([
                f'prediction_score_{horizon}',
                f'prediction_lower_ci_{horizon}',
                f'prediction_upper_ci_{horizon}'
            ])
        
        prediction_features.extend([
            'model_consensus_score', 'prediction_stability', 'ensemble_weight',
            'calibration_score', 'over_confident', 'under_confident'
        ])
        
        interpretability_features = [
            'age_importance', 'sex_importance', 'previous_attempts_importance',
            'mental_health_treatment_importance', 'depression_severity_importance',
            'social_support_importance', 'hopelessness_importance',
            'local_explanation_confidence', 'counterfactual_distance',
            'model_complexity_score', 'linear_separability', 'feature_interaction_strength',
            'clinical_rule_triggered', 'recommendation_type', 'epistemic_uncertainty',
            'aleatoric_uncertainty', 'total_uncertainty'
        ]
        
        return split_features + engineering_features + training_features + prediction_features + interpretability_features


class BiometricsDigitalGenerator(BaseFeatureGenerator):
    """
    Digital biomarkers and wearable device features
    Emerging field in digital health and precision medicine
    """
    
    def __init__(self):
        # Device adoption rates by age
        self.device_adoption = {
            'smartphone': 0.85,
            'fitness_tracker': 0.35,
            'smartwatch': 0.25,
            'sleep_tracker': 0.15,
            'mood_app': 0.20
        }
        
        # Digital biomarker categories
        self.biomarker_categories = [
            'sleep_patterns', 'activity_levels', 'social_interaction',
            'mood_indicators', 'stress_markers', 'cognitive_function'
        ]
    
    def generate_features(self, person: Dict, attempt_date: date, context: Dict) -> Dict:
        """Generate digital biomarker and wearable device features"""
        age = person['age']
        
        # Device ownership and usage
        device_features = self._generate_device_features(age)
        
        # Digital biomarkers
        biomarker_features = self._generate_digital_biomarkers(age, device_features)
        
        # App usage patterns
        app_usage_features = self._generate_app_usage_patterns(age)
        
        # Passive monitoring features
        passive_features = self._generate_passive_monitoring(device_features)
        
        return {
            **device_features,
            **biomarker_features,
            **app_usage_features,
            **passive_features
        }
    
    def _generate_device_features(self, age: int) -> Dict:
        """Generate device ownership and usage patterns"""
        features = {}
        
        for device, base_rate in self.device_adoption.items():
            # Age-adjusted adoption rates
            if age < 25:
                adjusted_rate = base_rate * 1.4  # Higher adoption in young adults
            elif age < 40:
                adjusted_rate = base_rate * 1.2
            elif age < 60:
                adjusted_rate = base_rate
            else:
                adjusted_rate = base_rate * 0.6  # Lower adoption in elderly
            
            adjusted_rate = min(0.95, adjusted_rate)  # Cap at 95%
            features[f'owns_{device}'] = np.random.random() < adjusted_rate
            
            # Usage intensity if device is owned
            if features[f'owns_{device}']:
                features[f'{device}_daily_usage_hours'] = round(np.random.gamma(2, 2), 1)
                features[f'{device}_engagement_score'] = round(np.random.beta(3, 2), 3)
            else:
                features[f'{device}_daily_usage_hours'] = 0.0
                features[f'{device}_engagement_score'] = 0.0
        
        # Overall digital engagement
        total_devices = sum(1 for device in self.device_adoption.keys() if features[f'owns_{device}'])
        features['digital_device_count'] = total_devices
        features['high_digital_engagement'] = total_devices >= 3
        
        return features
    
    def _generate_digital_biomarkers(self, age: int, device_features: Dict) -> Dict:
        """Generate digital biomarker measurements"""
        features = {}
        
        # Sleep patterns (if sleep tracker available)
        if device_features['owns_sleep_tracker'] or device_features['owns_smartwatch']:
            features['avg_sleep_duration_hours'] = round(np.random.normal(6.8, 1.2), 1)
            features['sleep_efficiency_percent'] = round(np.random.normal(82, 12), 1)
            features['sleep_latency_minutes'] = round(np.random.gamma(2, 15), 1)
            features['rem_sleep_percent'] = round(np.random.normal(22, 5), 1)
            features['sleep_regularity_score'] = round(np.random.beta(3, 2), 3)
            features['sleep_disruption_events'] = np.random.poisson(2)
        else:
            for key in ['avg_sleep_duration_hours', 'sleep_efficiency_percent', 
                       'sleep_latency_minutes', 'rem_sleep_percent', 
                       'sleep_regularity_score', 'sleep_disruption_events']:
                features[key] = None
        
        # Activity patterns (if fitness tracker available)
        if device_features['owns_fitness_tracker'] or device_features['owns_smartwatch']:
            features['daily_steps_avg'] = np.random.gamma(3, 2000)  # Average daily steps
            features['active_minutes_daily'] = round(np.random.gamma(3, 15), 1)
            features['sedentary_time_hours'] = round(np.random.normal(8.5, 2.0), 1)
            features['heart_rate_variability'] = round(np.random.normal(35, 10), 1)
            features['resting_heart_rate'] = round(np.random.normal(72, 12), 0)
            features['activity_consistency_score'] = round(np.random.beta(2, 3), 3)
        else:
            for key in ['daily_steps_avg', 'active_minutes_daily', 'sedentary_time_hours',
                       'heart_rate_variability', 'resting_heart_rate', 'activity_consistency_score']:
                features[key] = None
        
        # Smartphone-based biomarkers (if smartphone available)
        if device_features['owns_smartphone']:
            features['screen_time_hours_daily'] = round(np.random.gamma(4, 1.5), 1)
            features['app_switches_daily'] = np.random.poisson(120)
            features['typing_speed_wpm'] = round(np.random.normal(35, 8), 1)
            features['typing_irregularity'] = round(np.random.beta(2, 4), 3)
            features['call_duration_avg_minutes'] = round(np.random.gamma(2, 3), 1)
            features['social_app_usage_hours'] = round(np.random.gamma(2, 1), 1)
            features['location_entropy'] = round(np.random.beta(3, 3), 3)  # Movement predictability
        else:
            for key in ['screen_time_hours_daily', 'app_switches_daily', 'typing_speed_wpm',
                       'typing_irregularity', 'call_duration_avg_minutes', 'social_app_usage_hours',
                       'location_entropy']:
                features[key] = None
        
        return features
    
    def _generate_app_usage_patterns(self, age: int) -> Dict:
        """Generate mental health app usage patterns"""
        features = {}
        
        # Mental health app categories
        app_categories = {
            'meditation_mindfulness': 0.25,
            'mood_tracking': 0.15,
            'therapy_platforms': 0.10,
            'crisis_support': 0.08,
            'peer_support': 0.12,
            'cognitive_training': 0.08
        }
        
        for app_type, base_rate in app_categories.items():
            # Age adjustments
            if age < 30:
                adjusted_rate = base_rate * 1.5
            elif age < 50:
                adjusted_rate = base_rate
            else:
                adjusted_rate = base_rate * 0.7
            
            features[f'uses_{app_type}_app'] = np.random.random() < adjusted_rate
            
            if features[f'uses_{app_type}_app']:
                features[f'{app_type}_sessions_per_week'] = np.random.poisson(4)
                features[f'{app_type}_avg_session_minutes'] = round(np.random.gamma(2, 8), 1)
                features[f'{app_type}_adherence_rate'] = round(np.random.beta(3, 2), 3)
            else:
                features[f'{app_type}_sessions_per_week'] = 0
                features[f'{app_type}_avg_session_minutes'] = 0.0
                features[f'{app_type}_adherence_rate'] = 0.0
        
        # Overall digital mental health engagement
        total_apps = sum(1 for app in app_categories.keys() if features[f'uses_{app}_app'])
        features['mental_health_app_count'] = total_apps
        features['digital_therapy_engaged'] = total_apps >= 2
        
        return features
    
    def _generate_passive_monitoring(self, device_features: Dict) -> Dict:
        """Generate passive monitoring and ambient features"""
        features = {}
        
        # Communication patterns
        if device_features['owns_smartphone']:
            features['text_message_frequency_daily'] = np.random.poisson(25)
            features['call_frequency_daily'] = np.random.poisson(3)
            features['social_contact_diversity'] = np.random.poisson(8)  # Unique contacts
            features['response_time_minutes_avg'] = round(np.random.gamma(3, 20), 1)
            features['communication_regularity'] = round(np.random.beta(3, 2), 3)
        else:
            for key in ['text_message_frequency_daily', 'call_frequency_daily',
                       'social_contact_diversity', 'response_time_minutes_avg',
                       'communication_regularity']:
                features[key] = None
        
        # Digital behavior patterns
        features['notification_response_rate'] = round(np.random.beta(4, 3), 3)
        features['app_usage_fragmentation'] = round(np.random.beta(2, 3), 3)  # How scattered usage is
        features['digital_routine_consistency'] = round(np.random.beta(3, 2), 3)
        
        # Environmental sensors (if available)
        ambient_sensing_available = np.random.random() < 0.30  # 30% have ambient sensing
        if ambient_sensing_available:
            features['ambient_light_avg_lux'] = round(np.random.gamma(3, 100), 0)
            features['ambient_noise_avg_db'] = round(np.random.normal(45, 10), 1)
            features['temperature_avg_celsius'] = round(np.random.normal(22, 3), 1)
            features['air_quality_index'] = round(np.random.gamma(2, 25), 0)
        else:
            for key in ['ambient_light_avg_lux', 'ambient_noise_avg_db',
                       'temperature_avg_celsius', 'air_quality_index']:
                features[key] = None
        
        # Digital biomarker summary scores
        valid_biomarkers = sum(1 for key, value in features.items() 
                             if value is not None and 'score' in key)
        features['digital_biomarker_completeness'] = round(valid_biomarkers / 20, 2)  # Proportion of available biomarkers
        
        return features
    
    def get_feature_names(self) -> List[str]:
        device_features = []
        for device in self.device_adoption.keys():
            device_features.extend([
                f'owns_{device}',
                f'{device}_daily_usage_hours',
                f'{device}_engagement_score'
            ])
        device_features.extend(['digital_device_count', 'high_digital_engagement'])
        
        biomarker_features = [
            'avg_sleep_duration_hours', 'sleep_efficiency_percent', 'sleep_latency_minutes',
            'rem_sleep_percent', 'sleep_regularity_score', 'sleep_disruption_events',
            'daily_steps_avg', 'active_minutes_daily', 'sedentary_time_hours',
            'heart_rate_variability', 'resting_heart_rate', 'activity_consistency_score',
            'screen_time_hours_daily', 'app_switches_daily', 'typing_speed_wpm',
            'typing_irregularity', 'call_duration_avg_minutes', 'social_app_usage_hours',
            'location_entropy'
        ]
        
        app_features = []
        app_categories = ['meditation_mindfulness', 'mood_tracking', 'therapy_platforms',
                         'crisis_support', 'peer_support', 'cognitive_training']
        for app in app_categories:
            app_features.extend([
                f'uses_{app}_app',
                f'{app}_sessions_per_week',
                f'{app}_avg_session_minutes',
                f'{app}_adherence_rate'
            ])
        app_features.extend(['mental_health_app_count', 'digital_therapy_engaged'])
        
        passive_features = [
            'text_message_frequency_daily', 'call_frequency_daily', 'social_contact_diversity',
            'response_time_minutes_avg', 'communication_regularity', 'notification_response_rate',
            'app_usage_fragmentation', 'digital_routine_consistency', 'ambient_light_avg_lux',
            'ambient_noise_avg_db', 'temperature_avg_celsius', 'air_quality_index',
            'digital_biomarker_completeness'
        ]
        
        return device_features + biomarker_features + app_features + passive_features


# Enhanced Base Generator with Industry Standard Features
class IndustryStandardSurveillanceGenerator(BaseSurveillanceDataGenerator):
    """
    Enhanced surveillance generator with comprehensive industry-standard features
    """
    
    def __init__(self, start_year: int = 2023, end_year: int = 2027, 
                 random_seed: int = 42, feature_set: str = 'comprehensive'):
        """
        Initialize with selectable feature sets
        
        Parameters:
        - feature_set: 'basic', 'clinical', 'research', 'comprehensive'
        """
        super().__init__(start_year, end_year, random_seed)
        
        self.feature_set = feature_set
        self._initialize_industry_generators()
    
    def _initialize_industry_generators(self):
        """Initialize industry standard feature generators based on selected feature set"""
        
        if self.feature_set in ['clinical', 'research', 'comprehensive']:
            self.add_feature_generator(SociodemographicGenerator())
            self.add_feature_generator(ClinicalComorbidityGenerator())
            self.add_feature_generator(RiskFactorGenerator())
        
        if self.feature_set in ['research', 'comprehensive']:
            self.add_feature_generator(HealthcareSystemGenerator())
            self.add_feature_generator(MachineLearningGenerator())
        
        if self.feature_set == 'comprehensive':
            self.add_feature_generator(BiometricsDigitalGenerator())
    
    def generate_comprehensive_dataset(self) -> pd.DataFrame:
        """Generate dataset with industry standard features"""
        print(f"Generating comprehensive dataset with {self.feature_set} feature set...")
        
        dataset = self.generate_complete_dataset()
        
        # Add feature set metadata
        dataset['feature_set_type'] = self.feature_set
        dataset['generation_timestamp'] = datetime.now()
        
        # Calculate feature counts by category
        feature_counts = self._calculate_feature_counts(dataset)
        
        print(f"\nFeature Categories Generated:")
        for category, count in feature_counts.items():
            print(f"- {category}: {count} features")
        
        return dataset
    
    def _calculate_feature_counts(self, dataset: pd.DataFrame) -> Dict[str, int]:
        """Calculate number of features by category"""
        feature_counts = {}
        
        # Core surveillance features
        core_features = [col for col in dataset.columns if any(keyword in col.lower() 
                        for keyword in ['attempt_date', 'method', 'age', 'sex', 'followup'])]
        feature_counts['Core Surveillance'] = len(core_features)
        
        # Sociodemographic features
        socio_features = [col for col in dataset.columns if any(keyword in col.lower() 
                         for keyword in ['education', 'employment', 'marital', 'household', 'socioeconomic'])]
        feature_counts['Sociodemographic'] = len(socio_features)
        
        # Clinical features
        clinical_features = [col for col in dataset.columns if any(keyword in col.lower() 
                            for keyword in ['has_', 'prescribed_', 'gaf_', 'clinical_', 'comorbidity'])]
        feature_counts['Clinical'] = len(clinical_features)
        
        # Risk factors
        risk_features = [col for col in dataset.columns if any(keyword in col.lower() 
                        for keyword in ['risk_', 'protective_', 'hopelessness', 'depression_severity'])]
        feature_counts['Risk Factors'] = len(risk_features)
        
        # Healthcare system
        healthcare_features = [col for col in dataset.columns if any(keyword in col.lower() 
                              for keyword in ['care_', 'provider_', 'insurance_', 'accessed_'])]
        feature_counts['Healthcare System'] = len(healthcare_features)
        
        # Machine learning
        ml_features = [col for col in dataset.columns if any(keyword in col.lower() 
                      for keyword in ['split', 'prediction_', 'model_', 'importance', 'cv_fold'])]
        feature_counts['Machine Learning'] = len(ml_features)
        
        # Digital biomarkers
        digital_features = [col for col in dataset.columns if any(keyword in col.lower() 
                           for keyword in ['owns_', 'digital_', 'app_', 'sleep_', 'biomarker'])]
        feature_counts['Digital Biomarkers'] = len(digital_features)
        
        return feature_counts


# Usage example and demonstration
def demonstrate_industry_generators():
    """Demonstrate the comprehensive industry-standard generators"""
    
    print("="*80)
    print("INDUSTRY-STANDARD FEATURE GENERATORS DEMONSTRATION")
    print("="*80)
    
    # Initialize generator with comprehensive feature set
    generator = IndustryStandardSurveillanceGenerator(
        start_year=2023,
        end_year=2025,  # Shorter for demo
        random_seed=42,
        feature_set='comprehensive'
    )
    
    print(f"Initialized generator with {len(generator.feature_generators)} feature generators:")
    for i, gen in enumerate(generator.feature_generators):
        print(f"  {i+1}. {gen.__class__.__name__}")
    
    # Generate sample dataset
    print(f"\nGenerating sample dataset...")
    dataset = generator.generate_comprehensive_dataset()
    
    # Display feature summary
    print(f"\nDataset Summary:")
    print(f"- Total records: {len(dataset):,}")
    print(f"- Total features: {len(dataset.columns)}")
    print(f"- Date range: {dataset['attempt_date'].min()} to {dataset['attempt_date'].max()}")
    
    # Display sample of key features
    key_features = [
        'person_id', 'age_at_attempt', 'sex', 'method_primary',
        'education_level', 'employment_status', 'has_major_depression',
        'overall_risk_category', 'prediction_score_30_day', 'owns_smartphone'
    ]
    
    available_features = [f for f in key_features if f in dataset.columns]
    
    print(f"\nSample of key features:")
    print(dataset[available_features].head())
    
    return dataset


if __name__ == "__main__":
    # Run demonstration
    sample_dataset = demonstrate_industry_generators()
