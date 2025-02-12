from survey import analysis
import pandas as pd
import sys
from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")
sys.stdout = open(f'results/results-{date}.txt','wt')

# Load processed data
data = pd.read_csv('data/processed/data_encoded.csv')

# extract all criteria columns
criteria_cols = [col for col in data.columns if col.startswith('crit_')]
agenda_cols = [col for col in data.columns if col.startswith('goal_')]
occupation_cols = [col for col in data.columns if col.startswith('occ_')]
demographics = ['career_stage', 'primary_area']

def demographic_analysis_loop(data, demographic_col, criteria_cols):
    for criterion in criteria_cols:
        demographic_comparison = analysis.compare_demographic_criteria(
            data, 
            demographic_col=demographic_col,
            criterion_col=criterion
        )
        print(f"\nSelection rates by {demographic_col} for {criterion}:\n", demographic_comparison['selection_rates'])
        print(f"Effect size (Cramer's V): {demographic_comparison['effect_size']:.3f}")
        print(f"P-value: {demographic_comparison['p_value']:.3f}")


def demographic_intelligence_perception(data, demographic_col, intelligence_col):
    llm_perception = analysis.analyze_llm_perception_demographics(
        data,
        demographic_col=demographic_col,
        llm_col=intelligence_col
    )
    print("\nPerception breakdown:\n", llm_perception['perception_breakdown'])
    print(f"Demographic effect size: {llm_perception['effect_size']:.3f}")
    print(f"P-value: {llm_perception['p_value']:.3f}")


def analyze_research_agenda_llm(data, agenda_col, llm_col):
    agenda_analysis = analysis.analyze_research_agenda_llm(
        data,
        agenda_col=agenda_col,
        llm_col=llm_col
    )
    print("\nPerception by research agenda:\n", agenda_analysis['perception_breakdown'])
    print(f"Research agenda effect: {agenda_analysis['effect_size']:.3f}")
    print(f"P-value: {agenda_analysis['p_value']:.3f}")


def main():
    # 1. Analyze criteria correlations
    print("--- Criteria correlations ---")
    criteria_correlations = analysis.analyze_criteria_correlations(data, criteria_cols)

    print("Phi coefficients:\n", criteria_correlations['phi_coefficients'])
    print("\nSignificant correlations (p < 0.05):\n", 
        criteria_correlations['p_values'])
    
    # 2. Compare criteria selection based on career stage
    print("\n--- Criteria selection by career stage ---")
    for demo in demographics:
        print(f"\n--- {demo} ---")
        demographic_analysis_loop(data, demo, criteria_cols)

    # 3. Analyze LLM perception across demographics
    print("\n--- LLM perception across demographics ---")
    for demo in demographics:
        print(f"\n--- {demo} ---")
        demographic_intelligence_perception(data, demo, 'llm_intelligence')
    
    print("\n--- LLM perception across demographics for future intelligence ---")
    for demo in demographics:
         print("\n--- {demo} ---")
         demographic_intelligence_perception(data, demo, 'llm_intelligence_future')

    # 4. Analyze research agenda relationship with LLM perception
    print("\n--- Research agenda relationship with LLM perception ---")
    for agenda in agenda_cols:
        analyze_research_agenda_llm(data, agenda, 'llm_intelligence')
    
    print("\n--- Research agenda relationship with LLM perception for future intelligence ---")
    for agenda in agenda_cols:
        analyze_research_agenda_llm(data, agenda, 'llm_intelligence_future')


if __name__=="__main__":
    main()