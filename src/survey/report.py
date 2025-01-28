import pandas as pd
import json
import os
from datetime import datetime
from analyzer import IntelligenceResearchAnalysis

def export_json_results(results, output_dir='results/'):
    """Convert analysis results to JSON-serializable format and export."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types to Python types for JSON serialization
    json_results = {
        'criteria_agreement': {
            'frequencies': {k: float(v) for k, v in results['criteria_agreement']['frequencies'].to_dict().items()},
            'cooccurrence': results['criteria_agreement']['cooccurrence'].tolist(),
            'linkage': results['criteria_agreement']['linkage'].tolist()
        },
        'llm_perception': {
            'current_distribution': {str(k): int(v) for k, v in results['llm_perception']['current_distribution'].to_dict().items()},
            'future_distribution': {str(k): int(v) for k, v in results['llm_perception']['future_distribution'].to_dict().items()},
            'chi2_test': {
                'statistic': float(results['llm_perception']['chi2_test']['statistic']),
                'pvalue': float(results['llm_perception']['chi2_test']['pvalue'])
            },
            'demographic_patterns': {
                demo_var: {
                    'effect_size': float(pattern['effect_size']),
                    'fisher_pval': float(pattern['fisher_pval']),
                    'perception_breakdown': pattern['perception_breakdown'].to_dict(orient='index')
                }
                for demo_var, pattern in results['llm_perception']['demographic_patterns'].items()
            }
        },
        'criteria_correlations': {
            'phi_matrix': results['criteria_correlations']['phi_matrix'].to_dict(orient='index')
        },
        'demographic_criteria_selection': {
            demo_var: {
                'overall_tests': {
                    crit: {
                        'p_value': float(test['p_value']),
                        'effect_size': float(test['effect_size']),
                        'corrected_p_value': float(test['corrected_p_value']),
                        'significant': bool(test['significant'])
                    }
                    for crit, test in analysis['overall_tests'].items()
                },
                'significant_associations': [
                    {
                        'criterion': assoc['criterion'],
                        'effect_size': float(assoc['effect_size']),
                        'corrected_p_value': float(assoc['corrected_p_value']),
                        'selection_stats': {
                            str(group): {
                                'rate': float(stats['rate']),
                                'count': int(stats['count']),
                                'total': int(stats['total'])
                            }
                            for group, stats in assoc['selection_stats'].items()
                        },
                        'pairwise_comparisons': [
                            {
                                'group1': str(pair['group1']),
                                'group2': str(pair['group2']),
                                'p_value': float(pair['p_value']),
                                'corrected_p_value': float(pair['corrected_p_value']),
                                'effect_size': float(pair['effect_size']),
                                'group1_rate': float(pair['group1_rate']),
                                'group2_rate': float(pair['group2_rate'])
                            }
                            for pair in assoc['pairwise_comparisons']
                        ]
                    }
                    for assoc in analysis['significant_associations']
                ]
            }
            for demo_var, analysis in results['demographic_criteria_selection'].items()
        },
        'criteria_comparison': {
            'more_important': [
                {
                    'criterion': result['criterion'],
                    'importance': float(result['importance']),
                    'lacking': float(result['lacking']),
                    'difference': float(result['difference']),
                    'p_value': float(result['p_value'])
                }
                for result in results['criteria_comparison']['more_important']
            ],
            'more_lacking': [
                {
                    'criterion': result['criterion'],
                    'importance': float(result['importance']),
                    'lacking': float(result['lacking']),
                    'difference': float(result['difference']),
                    'p_value': float(result['p_value'])
                }
                for result in results['criteria_comparison']['more_lacking']
            ]
        },
        'multiple_testing': {
            'original_pvals': {k: float(v) for k, v in results['multiple_testing']['original_pvals'].items()},
            'corrected_pvals': {k: float(v) for k, v in results['multiple_testing']['corrected_pvals'].items()},
            'rejected_null': {k: bool(v) for k, v in results['multiple_testing']['rejected_null'].items()}
        }
    }
    
    # Export to JSON
    with open(f'{output_dir}/analysis_results_{timestamp}.json', 'w') as f:
        json.dump(json_results, f, indent=2)
        
    return json_results, timestamp

def generate_report(results, timestamp, output_dir='results/'):
    """Generate a readable report from analysis results."""
    with open(f'{output_dir}/summary_report_{timestamp}.txt', 'w') as f:
        f.write("INTELLIGENCE RESEARCH ANALYSIS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        # Summary Statistics
        f.write("0. SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        
        # Total respondents
        total_respondents = len(results['survey_data'])
        f.write(f"Total Respondents: {total_respondents}\n\n")
        
        # Career Stage Distribution
        f.write("Career Stage Distribution:\n")
        career_dist = results['survey_data']['career_stage'].value_counts()
        for stage, count in career_dist.items():
            percentage = (count / total_respondents) * 100
            f.write(f"  - {stage}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Primary Area Distribution
        f.write("Primary Research Area Distribution:\n")
        area_dist = results['survey_data']['primary_area'].value_counts()
        for area, count in area_dist.items():
            percentage = (count / total_respondents) * 100
            f.write(f"  - {area}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Occupation Distribution
        f.write("Occupation Distribution:\n")
        occ_counts = {}
        
        # Count individual occurrences in the occupation field
        for occs in results['survey_data']['occupation']:
            # Remove brackets and split the string
            occs = occs.strip("[]'").replace("'", "").split(", ")
            for occ in occs:
                occ_counts[occ] = occ_counts.get(occ, 0) + 1
        
        for occ, count in sorted(occ_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_respondents) * 100
            f.write(f"  - {occ}: {count} ({percentage:.1f}%)\n")
        f.write("\n")

        
        # Gender Distribution (if available)
        if 'gender' in results['survey_data'].columns:
            f.write("Gender Distribution:\n")
            gender_dist = results['survey_data']['gender'].value_counts()
            for gender, count in gender_dist.items():
                percentage = (count / total_respondents) * 100
                f.write(f"  - {gender}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
        
        f.write("\n")

        # Research Goals Distribution and Analysis
        f.write("1. RESEARCH GOALS ANALYSIS\n")
        f.write("-" * 20 + "\n")
        
        # Display frequencies
        f.write("Research Goal Distribution:\n")
        for goal, freq in results['research_goals']['frequencies'].items():
            f.write(f"  - {goal.replace('goal_', '')}: {freq:.1%}\n")
        f.write("\n")
        
        # Display strongest goal correlations
        f.write("Most Correlated Research Goals:\n")
        goal_matrix = results['research_goals']['cooccurrence']
        goal_cols = [col for col in results['research_goals']['frequencies'].index]
        
        # Get all pairs and their correlations
        goal_pairs = []
        for i in range(len(goal_matrix)):
            for j in range(i + 1, len(goal_matrix)):
                goal_pairs.append({
                    'pair': (goal_cols[i], goal_cols[j]),
                    'correlation': goal_matrix[i, j]
                })
        
        # Sort and display top correlations
        sorted_goal_pairs = sorted(goal_pairs, 
                                 key=lambda x: abs(x['correlation']), 
                                 reverse=True)
        
        for pair in sorted_goal_pairs[:5]:
            goal1 = pair['pair'][0].replace('goal_', '')
            goal2 = pair['pair'][1].replace('goal_', '')
            corr = pair['correlation']
            f.write(f"  - {goal1} & {goal2}: {corr:.3f}\n")
        f.write("\n")
        
        # Display strongest goal-criteria correlations
        f.write("Strongest Research Goal - Criteria Relationships:\n")
        goal_crit_matrix = results['research_goals']['goal_criteria_correlations']
        
        # Get all goal-criteria pairs and their correlations
        goal_crit_pairs = []
        for goal in goal_crit_matrix.index:
            for criterion in goal_crit_matrix.columns:
                corr = goal_crit_matrix.loc[goal, criterion]
                if abs(corr) > 0.2:  # Only show meaningful correlations
                    goal_crit_pairs.append({
                        'goal': goal,
                        'criterion': criterion,
                        'correlation': corr
                    })
        
        # Sort and display
        sorted_goal_crit = sorted(goal_crit_pairs, 
                                key=lambda x: abs(x['correlation']), 
                                reverse=True)
        
        for pair in sorted_goal_crit[:10]:
            goal = pair['goal'].replace('goal_', '')
            crit = pair['criterion'].replace('crit_', '')
            corr = pair['correlation']
            f.write(f"  - {goal} - {crit}: {corr:.3f}\n")
        f.write("\n")
        
        # Research Goals and LLM Perception
        f.write("Research Goals and LLM Perception:\n")
        for goal, stats in results['research_goals']['llm_perception'].items():
            if stats['effect_size'] > 0.15:  # Only show meaningful effects
                f.write(f"\n{goal.replace('goal_', '')}:\n")
                f.write(f"  Effect Size (Cramer's V): {stats['effect_size']:.3f}\n")
                f.write("  Perception Breakdown:\n")
                
                for has_goal in [1, 0]:
                    if has_goal in stats['perception_breakdown'].index:
                        f.write(f"    {'With' if has_goal else 'Without'} this goal:\n")
                        for perception, prop in stats['perception_breakdown'].loc[has_goal].items():
                            f.write(f"      - {perception}: {prop:.1%}\n")
        f.write("\n")
        
        # Criteria Agreement
        f.write("2. CRITERIA AGREEMENT\n")
        f.write("-" * 20 + "\n")
        f.write("Most Common Criteria:\n")
        for crit, freq in results['criteria_agreement']['frequencies'].nlargest(3).items():
            f.write(f"  - {crit.replace('crit_', '')}: {freq:.2%}\n")
        f.write("\n")

        # Add Criteria Correlations
        f.write("Most Correlated Criteria Pairs:\n")
        phi_matrix = results['criteria_correlations']['phi_matrix']
        
        # Get all pairs and their correlations
        corr_pairs = []
        for i in range(len(phi_matrix)):
            for j in range(i + 1, len(phi_matrix)):  # only upper triangle
                corr_pairs.append({
                    'pair': (phi_matrix.index[i], phi_matrix.index[j]),
                    'correlation': phi_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation
        sorted_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        
        # Top 5 strongest correlations
        f.write("Strongest Correlations:\n")
        for pair in sorted_pairs[:5]:
            crit1 = pair['pair'][0].replace('crit_', '')
            crit2 = pair['pair'][1].replace('crit_', '')
            corr = pair['correlation']
            f.write(f"  - {crit1} & {crit2}: {corr:.3f}\n")
        
        # Bottom 5 correlations
        f.write("\nWeakest Correlations:\n")
        for pair in sorted_pairs[-5:]:
            crit1 = pair['pair'][0].replace('crit_', '')
            crit2 = pair['pair'][1].replace('crit_', '')
            corr = pair['correlation']
            f.write(f"  - {crit1} & {crit2}: {corr:.3f}\n")
        f.write("\n")
        
        # LLM Perception Section
        f.write("3. LLM INTELLIGENCE PERCEPTION\n")
        f.write("-" * 20 + "\n")
        f.write("Current Perception:\n")
        for response, count in results['llm_perception']['current_distribution'].items():
            f.write(f"  - {response}: {count} responses\n")
        f.write("\nFuture Outlook:\n")
        for response, count in results['llm_perception']['future_distribution'].items():
            f.write(f"  - {response}: {count} responses\n")
        
        # Add demographic patterns for LLM perception
        f.write("\nDemographic Patterns in LLM Perception:\n")
        for demo_var, pattern in results['llm_perception']['demographic_patterns'].items():
            f.write(f"\n{demo_var.upper()}:\n")
            f.write(f"  Effect Size (Cramer's V): {pattern['effect_size']:.3f}\n")
            
            # Add perception breakdown by group
            f.write("  Perception by Group:\n")
            for group, perceptions in pattern['perception_breakdown'].to_dict(orient='index').items():
                f.write(f"    {group}:\n")
                for perception, prop in perceptions.items():
                    f.write(f"      - {perception}: {prop:.1%}\n")
        f.write("\n")
        
        # Demographic Criteria Selection
        f.write("4. DEMOGRAPHIC INFLUENCES ON CRITERIA SELECTION\n")
        f.write("-" * 20 + "\n")
        
        demo_criteria = results['demographic_criteria_selection']
        for demo_var, analysis in demo_criteria.items():
            f.write(f"\n{demo_var.upper()}:\n")
            
            sig_associations = sorted(
                analysis['significant_associations'],
                key=lambda x: x['effect_size'],
                reverse=True
            )
            
            if sig_associations:
                f.write("\nSignificant associations with criteria:\n")
                for assoc in sig_associations:
                    f.write(f"\n{assoc['criterion']}:\n")
                    f.write(f"  Effect size (Cramer's V): {assoc['effect_size']:.3f}\n")
                    f.write(f"  Corrected p-value: {assoc['corrected_p_value']:.4f}\n")
                    
                    # Selection rates
                    f.write("  Selection rates by group:\n")
                    for group, stats in assoc['selection_stats'].items():
                        f.write(f"    - {group}: {stats['rate']:.1%} ({stats['count']}/{stats['total']} respondents)\n")
                    
                    # Significant pairwise comparisons
                    sig_pairs = [p for p in assoc['pairwise_comparisons'] 
                               if p['corrected_p_value'] < 0.05]
                    if sig_pairs:
                        f.write("\n  Significant group differences (p < .05 after correction):\n")
                        for pair in sorted(sig_pairs, key=lambda x: x['effect_size'], reverse=True):
                            f.write(f"    - {pair['group1']} vs {pair['group2']}:\n")
                            f.write(f"      * Effect size (phi): {pair['effect_size']:.3f}\n")
                            f.write(f"      * Selection rates: {pair['group1_rate']:.1%} vs {pair['group2_rate']:.1%}\n")
                            f.write(f"      * p-value (corrected): {pair['corrected_p_value']:.4f}\n")
            else:
                f.write("\nNo significant associations found after correction.\n")
        
        # Criteria Importance vs Lacking Comparison
        f.write("\n5. CRITERIA IMPORTANCE VS LACKING COMPARISON\n")
        f.write("-" * 20 + "\n")
        
        # [Rest of the importance/lacking section remains the same]
        
        # Statistical Testing Summary
        f.write("\n6. STATISTICAL TESTING SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write("Significant results after correction:\n")
        for test, rejected in results['multiple_testing']['rejected_null'].items():
            if rejected:
                pval = results['multiple_testing']['corrected_pvals'][test]
                f.write(f"  - {test}: p = {pval:.4f}\n")

def main():
    """Main function to run the analysis and generate reports."""
    print("Running Intelligence Research Analysis...")
    
    # Load data
    survey_data = pd.read_csv('data/processed/data.csv')
    dummy_data = pd.read_csv('data/processed/data_encoded.csv')
    
    # Create analyzer and run analysis
    analyzer = IntelligenceResearchAnalysis(survey_data, dummy_data)
    results = analyzer.run_full_analysis()
    
    print("Analysis complete - generating report...")
    
    # Export results and generate report
    json_results, timestamp = export_json_results(results)
    generate_report(results, timestamp)
    
    print("Report generation complete!")

if __name__ == "__main__":
    main()