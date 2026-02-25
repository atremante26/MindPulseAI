from typing import List, Optional
from enum import Enum

class ResourceType(str, Enum):
    """Types of mental health resources."""
    APP = "app"
    THERAPY = "therapy"
    HOTLINE = "hotline"
    COMMUNITY = "community"
    SELF_HELP = "self_help"
    MEDICATION = "medication_info"
    LOCAL_SERVICE = "local_service"

class CostTier(str, Enum):
    """Cost tiers for resources."""
    FREE = "free"
    LOW = "low" # $0-50/month
    MEDIUM = "medium" # $50-150/month
    HIGH = "high" # $150+/month

class Concern(str, Enum):
    """Mental health concerns that resources address."""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    PANIC_ATTACKS = "panic_attacks"
    PTSD = "ptsd"
    ADHD = "adhd"
    BIPOLAR = "bipolar"
    EATING_DISORDER = "eating_disorder"
    SUBSTANCE_ABUSE = "substance_abuse"
    SELF_HARM = "self_harm"
    SUICIDAL_THOUGHTS = "suicidal_thoughts"
    STRESS = "stress"
    LONELINESS = "loneliness"
    RELATIONSHIP_ISSUES = "relationship_issues"
    GRIEF = "grief"
    TRAUMA = "trauma"

class AgeGroup(str, Enum):
    """Age groups for targeted resources."""
    TEEN = "teen" # 13-17
    YOUNG_ADULT = "young_adult" # 18-25
    ADULT = "adult" # 26-64
    SENIOR = "senior" # 65+
    ALL = "all"

# Resource data structure
RESOURCE_SCHEMA = {
    "id": str,  # Unique identifier
    "name": str,  # Resource name
    "type": ResourceType,  # Type of resource
    "description": str,  # Brief description
    "url": Optional[str],  # Website URL
    "phone": Optional[str],  # Phone number (for hotlines)
    "concerns": List[Concern],  # What it helps with
    "cost_tier": CostTier,  # Cost level
    "cost_details": Optional[str],  # "Free for first month", "$99/month"
    "age_groups": List[AgeGroup],  # Target ages
    "availability": str,  # "24/7", "Business hours", "By appointment"
    "online_only": bool,  # True if no physical location needed
    "crisis_resource": bool,  # True if for emergencies
    "requires_insurance": bool,  # True if insurance needed
    "languages": List[str],  # Supported languages
    "rating": Optional[float],  # User rating (1-5), if available
    "tags": List[str],  # Additional tags for matching
} 