import sys
import logging
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_data import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_who_suicide(who_df):
    """
    Analyze WHO suicide dataset and return key statistics.
    Columns: country, year, sex, age, suicides_no, population, suicide_rate_per_100k
    """
    try:
        # Get most recent year
        latest_year = who_df['year'].max()
        latest_data = who_df[who_df['year'] == latest_year].copy()
        
        # Global totals (latest year)
        total_suicides = latest_data['suicides_no'].sum()
        global_rate = latest_data['suicide_rate_per_100k'].mean()
        
        # Sex disparity
        sex_stats = latest_data.groupby('sex').agg({
            'suicides_no': 'sum',
            'population': 'sum',
            'suicide_rate_per_100k': 'mean'
        })
        male_rate = sex_stats.loc['male', 'suicide_rate_per_100k']
        female_rate = sex_stats.loc['female', 'suicide_rate_per_100k']
        sex_ratio = male_rate / female_rate if female_rate > 0 else 0
        
        # Age group analysis (highest risk)
        age_stats = latest_data.groupby('age').agg({
            'suicides_no': 'sum',
            'suicide_rate_per_100k': 'mean'
        }).sort_values('suicide_rate_per_100k', ascending=False)
        
        highest_risk_age = age_stats.index[0]
        highest_risk_rate = age_stats['suicide_rate_per_100k'].iloc[0]
        
        # Top 5 countries by rate
        country_stats = latest_data.groupby('country').agg({
            'suicide_rate_per_100k': 'mean'
        }).sort_values('suicide_rate_per_100k', ascending=False)
        
        top_5_countries = country_stats.head(5).to_dict()['suicide_rate_per_100k']
        
        # 10-year trend
        ten_years_ago = latest_year - 10
        if ten_years_ago >= who_df['year'].min():
            trend_data = who_df[who_df['year'].isin([ten_years_ago, latest_year])]
            trend_totals = trend_data.groupby('year')['suicides_no'].sum()
            
            if len(trend_totals) == 2:
                old_total = trend_totals.iloc[0]
                new_total = trend_totals.iloc[1]
                trend_pct_change = ((new_total - old_total) / old_total) * 100
                trend_direction = "increasing" if trend_pct_change > 0 else "decreasing"
            else:
                trend_pct_change = 0
                trend_direction = "stable"
        else:
            trend_pct_change = 0
            trend_direction = "insufficient data"
        
        stats = {
            'year': latest_year,
            'global_suicides': int(total_suicides),
            'global_rate': round(global_rate, 1),
            'male_rate': round(male_rate, 1),
            'female_rate': round(female_rate, 1),
            'sex_ratio': round(sex_ratio, 1),
            'highest_risk_age_group': highest_risk_age,
            'highest_risk_rate': round(highest_risk_rate, 1),
            'top_5_countries': {k: round(v, 1) for k, v in list(top_5_countries.items())[:5]},
            'trend_direction': trend_direction,
            'trend_pct_change': round(trend_pct_change, 1),
            'trend_years': f"{ten_years_ago}-{latest_year}"
        }
        
        logger.info(f"Analyzed WHO suicide data: {latest_year}, {total_suicides:,.0f} suicides globally")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to analyze WHO suicide data: {e}")
        return None


def analyze_mental_health_care(mhc_df):
    """
    Analyze mental health care dataset and return key statistics.
    Columns: Indicator, Group, State, Subgroup, Phase, Time Period, Value, etc.
    """
    try:
        # Get unique indicators to understand what data we have
        indicators = mhc_df['Indicator'].unique()
        
        # Therapy/Counseling rates
        therapy_data = mhc_df[mhc_df['Indicator'].str.contains('Counseling|Therapy', case=False, na=False)]
        if not therapy_data.empty:
            # National average (most recent period)
            national_therapy = therapy_data[
                (therapy_data['State'] == 'United States') & 
                (therapy_data['Subgroup'] == 'United States')
            ]
            if not national_therapy.empty:
                latest_therapy = national_therapy.nlargest(1, 'Time Period')
                therapy_rate = latest_therapy['Value'].iloc[0] if not latest_therapy.empty else None
            else:
                therapy_rate = therapy_data['Value'].mean()
        else:
            therapy_rate = None
        
        # Medication rates
        medication_data = mhc_df[mhc_df['Indicator'].str.contains('Medication|Prescription', case=False, na=False)]
        if not medication_data.empty:
            national_meds = medication_data[
                (medication_data['State'] == 'United States') & 
                (medication_data['Subgroup'] == 'United States')
            ]
            if not national_meds.empty:
                latest_meds = national_meds.nlargest(1, 'Time Period')
                medication_rate = latest_meds['Value'].iloc[0] if not latest_meds.empty else None
            else:
                medication_rate = medication_data['Value'].mean()
        else:
            medication_rate = None
        
        # Unmet need / barriers (if available)
        unmet_need_data = mhc_df[mhc_df['Indicator'].str.contains('Unmet|Need|Barrier', case=False, na=False)]
        if not unmet_need_data.empty:
            unmet_need_rate = unmet_need_data['Value'].mean()
        else:
            unmet_need_rate = None
        
        # State-level variation (top and bottom 5 states by treatment access)
        # Use therapy data if available
        if not therapy_data.empty:
            state_therapy = therapy_data[
                (therapy_data['State'] != 'United States') &
                (therapy_data['Subgroup'] == therapy_data['State'])  # State-level only
            ]
            if not state_therapy.empty:
                # Get most recent period per state
                latest_by_state = state_therapy.loc[state_therapy.groupby('State')['Time Period'].idxmax()]
                state_avg = latest_by_state.groupby('State')['Value'].mean().sort_values()
                
                lowest_5_states = state_avg.head(5).to_dict()
                highest_5_states = state_avg.tail(5).to_dict()
            else:
                lowest_5_states = {}
                highest_5_states = {}
        else:
            lowest_5_states = {}
            highest_5_states = {}
        
        # Demographics (by sex, if available)
        sex_data = therapy_data[therapy_data['Group'] == 'By Sex'] if not therapy_data.empty else pd.DataFrame()
        if not sex_data.empty:
            sex_rates = sex_data.groupby('Subgroup')['Value'].mean().to_dict()
        else:
            sex_rates = {}
        
        stats = {
            'therapy_counseling_rate': round(therapy_rate, 1) if therapy_rate else None,
            'medication_rate': round(medication_rate, 1) if medication_rate else None,
            'unmet_need_rate': round(unmet_need_rate, 1) if unmet_need_rate else None,
            'lowest_access_states': {k: round(v, 1) for k, v in lowest_5_states.items()},
            'highest_access_states': {k: round(v, 1) for k, v in highest_5_states.items()},
            'treatment_by_sex': {k: round(v, 1) for k, v in sex_rates.items()},
            'available_indicators': list(indicators)[:5]  # First 5 indicators
        }
        
        logger.info(f"Analyzed mental health care data: {len(indicators)} indicators")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to analyze mental health care data: {e}")
        return None


def analyze_suicide_demographics(demo_df):
    """
    Analyze suicide demographics dataset and return key statistics. 
    Columns: indicator, unit, stub_name, stub_label, year, age, estimate, flag, demographic_category, demographic_value
    """
    try:
        # Get most recent year
        latest_year = demo_df['year'].max()
        latest_data = demo_df[demo_df['year'] == latest_year].copy()
        
        # Overall rate (Total/All persons)
        overall = latest_data[
            (latest_data['demographic_category'] == 'Total') &
            (latest_data['demographic_value'] == 'All persons')
        ]
        overall_rate = overall['estimate'].mean() if not overall.empty else None
        
        # By age group
        age_data = latest_data[latest_data['demographic_category'] == 'Age']
        if not age_data.empty:
            age_rates = age_data.groupby('demographic_value')['estimate'].mean().sort_values(ascending=False)
            age_breakdown = age_rates.to_dict()
            highest_risk_age = age_rates.index[0] if len(age_rates) > 0 else None
            highest_age_rate = age_rates.iloc[0] if len(age_rates) > 0 else None
        else:
            age_breakdown = {}
            highest_risk_age = None
            highest_age_rate = None
        
        # By sex/gender
        sex_data = latest_data[latest_data['demographic_category'] == 'Sex']
        if not sex_data.empty:
            sex_rates = sex_data.groupby('demographic_value')['estimate'].mean()
            sex_breakdown = sex_rates.to_dict()
            
            if 'Male' in sex_breakdown and 'Female' in sex_breakdown:
                sex_ratio = sex_breakdown['Male'] / sex_breakdown['Female']
            else:
                sex_ratio = None
        else:
            sex_breakdown = {}
            sex_ratio = None
        
        # By race/ethnicity
        race_data = latest_data[latest_data['demographic_category'] == 'Race and Hispanic origin']
        if not race_data.empty:
            race_rates = race_data.groupby('demographic_value')['estimate'].mean().sort_values(ascending=False)
            race_breakdown = race_rates.to_dict()
        else:
            race_breakdown = {}
        
        # 10-year trend
        ten_years_ago = latest_year - 10
        if ten_years_ago >= demo_df['year'].min():
            trend_data = demo_df[
                (demo_df['year'].isin([ten_years_ago, latest_year])) &
                (demo_df['demographic_category'] == 'Total')
            ]
            if not trend_data.empty:
                trend_rates = trend_data.groupby('year')['estimate'].mean()
                if len(trend_rates) == 2:
                    old_rate = trend_rates.iloc[0]
                    new_rate = trend_rates.iloc[1]
                    trend_pct_change = ((new_rate - old_rate) / old_rate) * 100
                    trend_direction = "increasing" if trend_pct_change > 0 else "decreasing"
                else:
                    trend_pct_change = 0
                    trend_direction = "stable"
            else:
                trend_pct_change = 0
                trend_direction = "insufficient data"
        else:
            trend_pct_change = 0
            trend_direction = "insufficient data"
        
        stats = {
            'year': latest_year,
            'overall_rate': round(overall_rate, 1) if overall_rate else None,
            'age_breakdown': {k: round(v, 1) for k, v in age_breakdown.items()},
            'highest_risk_age_group': highest_risk_age,
            'highest_age_rate': round(highest_age_rate, 1) if highest_age_rate else None,
            'sex_breakdown': {k: round(v, 1) for k, v in sex_breakdown.items()},
            'sex_ratio_male_to_female': round(sex_ratio, 1) if sex_ratio else None,
            'race_breakdown': {k: round(v, 1) for k, v in race_breakdown.items()},
            'trend_direction': trend_direction,
            'trend_pct_change': round(trend_pct_change, 1),
            'trend_years': f"{ten_years_ago}-{latest_year}"
        }
        
        logger.info(f"Analyzed suicide demographics: {latest_year}, overall rate {overall_rate}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to analyze suicide demographics: {e}")
        return None


def format_context_from_stats(who_stats, mhc_stats, demo_stats):
    """
    Convert analyzed statistics into formatted text for LLM context.
    
    Parameters:
    - who_stats: dict from analyze_who_suicide()
    - mhc_stats: dict from analyze_mental_health_care()
    - demo_stats: dict from analyze_suicide_demographics()
    
    Returns:
    - Formatted string for LLM prompt
    """
    context_parts = []
    
    # WHO Global Statistics
    if who_stats:
        context_parts.append(f"""
**WHO Global Suicide Statistics ({who_stats['year']}):**
- **Global deaths:** {who_stats['global_suicides']:,} suicides annually
- **Global rate:** {who_stats['global_rate']} per 100,000 population
- **Sex disparity:** Males {who_stats['sex_ratio']}x more likely (Male: {who_stats['male_rate']}, Female: {who_stats['female_rate']} per 100k)
- **Highest risk age group:** {who_stats['highest_risk_age_group']} ({who_stats['highest_risk_rate']} per 100,000)
- **10-year trend:** {who_stats['trend_direction']} ({who_stats['trend_pct_change']:+.1f}% change {who_stats['trend_years']})
- **Highest rate countries:** {', '.join([f"{country} ({rate})" for country, rate in list(who_stats['top_5_countries'].items())[:3]])}
""")
    
    # Mental Health Care
    if mhc_stats:
        therapy_text = f"{mhc_stats['therapy_counseling_rate']}% received therapy/counseling" if mhc_stats['therapy_counseling_rate'] else "N/A"
        medication_text = f"{mhc_stats['medication_rate']}% taking psychiatric medication" if mhc_stats['medication_rate'] else "N/A"
        
        context_parts.append(f"""
**US Mental Health Care Access:**
- **Therapy/counseling:** {therapy_text}
- **Medication use:** {medication_text}
- **Unmet need:** {mhc_stats['unmet_need_rate']}% if mhc_stats['unmet_need_rate'] else "Data unavailable"
- **Lowest access states:** {', '.join([f"{state} ({rate}%)" for state, rate in list(mhc_stats['lowest_access_states'].items())[:3]])} if mhc_stats['lowest_access_states'] else "N/A"
- **Highest access states:** {', '.join([f"{state} ({rate}%)" for state, rate in list(mhc_stats['highest_access_states'].items())[:3]])} if mhc_stats['highest_access_states'] else "N/A"
""")
    
    # Suicide Demographics
    if demo_stats:
        age_list = '\n'.join([f"  - {age}: {rate} per 100,000" for age, rate in list(demo_stats['age_breakdown'].items())[:5]])
        sex_list = '\n'.join([f"  - {sex}: {rate} per 100,000" for sex, rate in demo_stats['sex_breakdown'].items()])
        race_list = '\n'.join([f"  - {race}: {rate} per 100,000" for race, rate in list(demo_stats['race_breakdown'].items())[:5]])
        
        context_parts.append(f"""
**US Suicide Demographics ({demo_stats['year']}):**
- **Overall rate:** {demo_stats['overall_rate']} per 100,000 population
- **Sex disparity:** Males {demo_stats['sex_ratio_male_to_female']}x more likely than females
- **By sex:**
{sex_list}
- **By age group:**
{age_list}
- **By race/ethnicity:**
{race_list}
- **10-year trend:** {demo_stats['trend_direction']} ({demo_stats['trend_pct_change']:+.1f}% change {demo_stats['trend_years']})
""")
    return '\n'.join(context_parts)


def get_static_context():
    """
    Load static datasets (WHO Suicide, Mental Health Care, Suicide Demographics), analyze them, and return formatted context for LLM prompts.
    """
    try:
        
        logger.info("Loading static datasets for context...")
        
        # Load datasets
        who_suicide = load_dataset("who_suicide")
        mental_health_care = load_dataset("mental_health_care")
        suicide_demographics = load_dataset("suicide_demographics")
        
        logger.info("Successfully loaded all static datasets")
        
        # Analyze each dataset
        who_stats = analyze_who_suicide(who_suicide)
        mhc_stats = analyze_mental_health_care(mental_health_care)
        demo_stats = analyze_suicide_demographics(suicide_demographics)
        
        # Format into context text
        context = format_context_from_stats(who_stats, mhc_stats, demo_stats)
        
        logger.info("Generated static context from datasets")
        return context
        
    except Exception as e:
        logger.warning(f"Failed to load static datasets, using fallback context: {e}")
        
        # Fallback context if datasets unavailable
        return None


def get_crisis_indicators():
    """Returns list of crisis-related keywords to flag."""
    return [
        'suicide', 'suicidal', 'self-harm', 'self harm', 'cutting',
        'ending it all', 'want to die', 'no reason to live',
        'better off dead', 'kill myself', 'overdose'
    ]


def get_common_concerns():
    """
    Returns common mental health concerns to look for in data.
    Useful for thematic analysis.
    """
    return {
        'depression': [
            'depressed', 'depression', 'hopeless', 'hopelessness',
            'sad', 'empty', 'numb', 'worthless'
        ],
        'anxiety': [
            'anxiety', 'anxious', 'panic', 'panic attack', 
            'worry', 'nervous', 'overwhelmed', 'stressed'
        ],
        'therapy_access': [
            'therapy', 'therapist', 'counselor', 'psychiatrist',
            'afford', 'cost', 'expensive', 'insurance', 
            'wait list', 'waitlist', 'no availability'
        ],
        'medication': [
            'medication', 'meds', 'antidepressant', 'ssri', 
            'prescription', 'side effects', 'dosage', 'lexapro', 
            'zoloft', 'prozac', 'wellbutrin'
        ],
        'loneliness': [
            'lonely', 'alone', 'isolated', 'isolation',
            'no friends', 'social isolation', 'nobody cares'
        ],
        'work_stress': [
            'burnout', 'work stress', 'job stress', 'career',
            'workplace', 'overworked', 'toxic work environment'
        ],
        'relationships': [
            'breakup', 'divorce', 'relationship problems',
            'family issues', 'toxic relationship', 'abuse', 'domestic violence'
        ],
        'self_care': [
            'self-care', 'self care', 'coping', 'meditation',
            'exercise', 'sleep', 'routine', 'mindfulness'
        ],
        'crisis': [
            'crisis', 'emergency', 'urgent', '988', 'hotline',
            'hospital', 'ER', 'immediate help'
        ]
    }