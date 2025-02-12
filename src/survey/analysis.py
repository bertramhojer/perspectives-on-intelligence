import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

def analyze_criteria_correlations(data, criteria_cols):
    """
    Analyze correlations between criteria using phi coefficients.
    
    Args:
        data (pd.DataFrame): DataFrame with binary criteria columns (dummy encoded)
        criteria_cols (list): List of criteria column names
    
    Returns:
        dict: Contains phi coefficients matrix and significance tests
    """
    n_criteria = len(criteria_cols)
    phi_matrix = np.zeros((n_criteria, n_criteria))
    p_values = np.zeros((n_criteria, n_criteria))
    
    for i, col1 in enumerate(criteria_cols):
        for j, col2 in enumerate(criteria_cols):
            if i < j:  # Only calculate upper triangle
                contingency = pd.crosstab(data[col1], data[col2])
                chi2, p_val = stats.chi2_contingency(contingency)[:2]
                n = contingency.sum().sum()
                phi = np.sqrt(chi2 / n)
                
                # Adjust sign based on correlation
                if np.corrcoef(data[col1], data[col2])[0,1] < 0:
                    phi = -phi
                    
                phi_matrix[i,j] = phi
                phi_matrix[j,i] = phi
                p_values[i,j] = p_val
                p_values[j,i] = p_val
    
    return {
        'phi_coefficients': pd.DataFrame(phi_matrix, index=criteria_cols, columns=criteria_cols),
        'p_values': pd.DataFrame(p_values, index=criteria_cols, columns=criteria_cols)
    }

def compare_demographic_criteria(data, demographic_col, criterion_col):
    """
    Compare criterion selection between demographic groups.
    
    Args:
        data (pd.DataFrame): DataFrame with demographic and criterion columns
        demographic_col (str): Name of demographic column
        criterion_col (str): Name of criterion column
        
    Returns:
        dict: Contains contingency table, test statistics and effect size
    """
    contingency = pd.crosstab(data[demographic_col], data[criterion_col])
    
    # Fisher's exact test
    _, p_val = stats.fisher_exact(contingency)
    
    # Effect size (Cramer's V)
    chi2, _ = stats.chi2_contingency(contingency)[:2]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Selection rates per group
    rates = data.groupby(demographic_col)[criterion_col].mean()
    
    return {
        'contingency': contingency,
        'p_value': p_val,
        'effect_size': cramer_v,
        'selection_rates': rates
    }

def analyze_llm_perception_demographics(data, demographic_col, llm_col):
    """
    Analyze differences in LLM intelligence perception across demographic groups.
    
    Args:
        data (pd.DataFrame): DataFrame with demographic and LLM perception columns
        demographic_col (str): Name of demographic column
        llm_col (str): Name of LLM perception column
        
    Returns:
        dict: Contains test statistics and perception breakdown
    """
    contingency = pd.crosstab(data[demographic_col], data[llm_col])
    
    # Fisher's exact test
    _, p_val = stats.fisher_exact(contingency)
    
    # Effect size (Cramer's V)
    chi2, _ = stats.chi2_contingency(contingency)[:2]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Perception breakdown by group
    perception_by_group = pd.crosstab(
        data[demographic_col], 
        data[llm_col], 
        normalize='index'
    )

    try:
        perception_by_group["positive"] = perception_by_group["Agree"] + perception_by_group["Strongly agree"]
        perception_by_group["negative"] = perception_by_group["Disagree"] + perception_by_group["Strongly disagree"]
    except KeyError as e:
        perception_by_group["positive"] = perception_by_group["Yes, I agree"] + perception_by_group["Yes, I strongly agree"]
        perception_by_group["negative"] = perception_by_group["No, I disagree"] + perception_by_group["No, I strongly disagree"]
    
    return {
        'contingency': contingency,
        'p_value': p_val,
        'effect_size': cramer_v,
        'perception_breakdown': perception_by_group
    }

def analyze_research_agenda_llm(data, agenda_col, llm_col):
    """
    Analyze relationship between research agenda and LLM intelligence perception.
    
    Args:
        data (pd.DataFrame): DataFrame with research agenda and LLM perception columns
        agenda_col (str): Name of research agenda column
        llm_col (str): Name of LLM perception column
        
    Returns:
        dict: Contains test statistics and perception breakdown
    """
    contingency = pd.crosstab(data[agenda_col], data[llm_col])
    
    # Fisher's exact test
    _, p_val = stats.fisher_exact(contingency)
    
    # Effect size (Cramer's V)
    chi2, _ = stats.chi2_contingency(contingency)[:2]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Perception breakdown by agenda
    perception_by_agenda = pd.crosstab(
        data[agenda_col], 
        data[llm_col], 
        normalize='index'
    )

    try:
        perception_by_agenda["positive"] = perception_by_agenda["Agree"] + perception_by_agenda["Strongly agree"]
        perception_by_agenda["negative"] = perception_by_agenda["Disagree"] + perception_by_agenda["Strongly disagree"]
    except KeyError as e:
        perception_by_agenda["positive"] = perception_by_agenda["Yes, I agree"] + perception_by_agenda["Yes, I strongly agree"]
        perception_by_agenda["negative"] = perception_by_agenda["No, I disagree"] + perception_by_agenda["No, I strongly disagree"]
    
    return {
        'contingency': contingency,
        'p_value': p_val,
        'effect_size': cramer_v,
        'perception_breakdown': perception_by_agenda
    }
