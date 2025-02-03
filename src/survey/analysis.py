import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prince import MCA  # for multiple correspondence analysis
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class IntelligenceResearchAnalysis:
    def __init__(self, survey_data, dummy_data):
        """Initialize with both raw and dummy-encoded survey data."""
        self.survey_data = survey_data
        self.dummy_data = dummy_data
        self.results = {}
        
    def analyze_criteria_agreement(self):
        """Analyze agreement about key criteria for intelligence."""
        # Get criteria columns
        criteria_cols = [col for col in self.dummy_data.columns if col.startswith('crit_')]
        
        # Calculate criteria frequencies
        frequencies = self.dummy_data[criteria_cols].sum() / len(self.dummy_data)
        
        # Calculate co-occurrence matrix (essentially pearson correlation (phi-coefficient for binary data))
        cooccurrence = np.corrcoef(self.dummy_data[criteria_cols].T)
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(cooccurrence, method='ward')
        
        # Store results
        self.results['criteria_agreement'] = {
            'frequencies': frequencies,
            'cooccurrence': cooccurrence,
            'linkage': linkage,
            'criteria_list': criteria_cols
        }
        
    def analyze_llm_perception(self):
        """Analyze perception of LLM intelligence, including demographic patterns."""
        # Calculate overall distribution of responses
        llm_dist = self.survey_data['llm_intelligence'].value_counts()
        future_dist = self.survey_data['llm_intelligence_future'].value_counts()
        
        # Analyze relationship between lacking criteria and intelligence assessment
        lacking_cols = [col for col in self.dummy_data.columns if col.startswith('lack_')]
        lacking_by_perception = pd.crosstab(
            self.survey_data['llm_intelligence'],
            self.dummy_data[lacking_cols].sum(axis=1)
        )
        
        # Chi-square test for overall distribution
        chi2, pval = stats.chisquare(llm_dist)
        
        # Analyze demographic patterns in LLM perception
        demographic_vars = ['career_stage', 'primary_area', 'occupation']
        demographic_patterns = {}
        
        for demo_var in demographic_vars:
            # Create contingency table
            contingency = pd.crosstab(
                self.survey_data[demo_var],
                self.survey_data['llm_intelligence']
            )
            
            # Fisher's exact test
            _, pval = stats.fisher_exact(contingency)
            
            # Cramer's V effect size
            chi2, _, _, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim))
            
            # Calculate proportions for each group
            perception_by_demo = pd.crosstab(
                self.survey_data[demo_var],
                self.survey_data['llm_intelligence'],
                normalize='index'
            )
            
            demographic_patterns[demo_var] = {
                'contingency': contingency,
                'fisher_pval': pval,
                'effect_size': cramer_v,
                'perception_breakdown': perception_by_demo
            }
        
        self.results['llm_perception'] = {
            'current_distribution': llm_dist,
            'future_distribution': future_dist,
            'lacking_crosstab': lacking_by_perception,
            'chi2_test': {'statistic': chi2, 'pvalue': pval},
            'demographic_patterns': demographic_patterns
        }
        

    def analyze_criteria_correlations(self):
        """Analyze correlations between different criteria."""
        criteria_cols = [col for col in self.dummy_data.columns if col.startswith('crit_')]
        
        # Calculate phi coefficients
        phi_matrix = np.zeros((len(criteria_cols), len(criteria_cols)))
        
        for i, col1 in enumerate(criteria_cols):
            for j, col2 in enumerate(criteria_cols):
                if i != j:
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.dummy_data[col1],
                        self.dummy_data[col2]
                    )
                    
                    # Calculate chi-square
                    chi2, _ = stats.chi2_contingency(contingency)[:2]
                    
                    # Calculate phi coefficient
                    n = contingency.sum().sum()
                    phi = np.sqrt(chi2 / n)
                    
                    # Adjust sign based on correlation
                    if stats.pearsonr(self.dummy_data[col1], self.dummy_data[col2])[0] < 0:
                        phi = -phi
                    
                    phi_matrix[i, j] = phi
                
        self.results['criteria_correlations'] = {
            'phi_matrix': pd.DataFrame(
                phi_matrix,
                index=criteria_cols,
                columns=criteria_cols
            )
        }


    
    def pairwise_group_comparisons(self, demo_var, criterion):
        """Perform pairwise comparisons between demographic groups for a criterion."""
        groups = self.survey_data[demo_var].unique()
        pairwise_results = []
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                # Select only these two groups
                mask = self.survey_data[demo_var].isin([group1, group2])
                subtable = pd.crosstab(
                    self.survey_data.loc[mask, demo_var],
                    self.dummy_data.loc[mask, criterion]
                )
                
                # Fisher's exact test
                _, pval = stats.fisher_exact(subtable)
                
                # Effect size (phi coefficient for 2x2)
                chi2, _ = stats.chi2_contingency(subtable)[:2]
                n = subtable.sum().sum()
                phi = np.sqrt(chi2 / n)
                
                # Get selection rates for each group
                group1_rate = self.dummy_data[criterion][self.survey_data[demo_var] == group1].mean()
                group2_rate = self.dummy_data[criterion][self.survey_data[demo_var] == group2].mean()
                
                pairwise_results.append({
                    'group1': group1,
                    'group2': group2,
                    'p_value': pval,
                    'effect_size': phi,
                    'group1_rate': group1_rate,
                    'group2_rate': group2_rate
                })
        
        # Correct for multiple comparisons within this criterion
        pvals = [r['p_value'] for r in pairwise_results]
        _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
        
        for result, corrected_pval in zip(pairwise_results, pvals_corrected):
            result['corrected_p_value'] = corrected_pval
        
        return pairwise_results

    def analyze_demographic_criteria_selection(self):
        """Analyze how demographics influence criteria selection with proper statistical controls."""
        demographic_vars = ['career_stage', 'primary_area', 'occupation', 'research_goals']
        criteria_cols = [col for col in self.dummy_data.columns if col.startswith('crit_')]
        
        demo_criteria_results = {}
        
        for demo_var in demographic_vars:
            demo_results = {
                'overall_tests': {},
                'significant_associations': []
            }
            
            for criterion in criteria_cols:
                # Overall test
                contingency = pd.crosstab(
                    self.survey_data[demo_var],
                    self.dummy_data[criterion]
                )
                
                # Fisher's exact test
                _, pval = stats.fisher_exact(contingency)
                
                # Calculate Cramer's V
                chi2, _ = stats.chi2_contingency(contingency)[:2]
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramer_v = np.sqrt(chi2 / (n * min_dim))
                
                demo_results['overall_tests'][criterion.replace('crit_', '')] = {
                    'p_value': pval,
                    'effect_size': cramer_v
                }
            
            # Correct for multiple comparisons across criteria
            pvals = [test['p_value'] for test in demo_results['overall_tests'].values()]
            rejected, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
            
            for (criterion, test), corrected_pval, is_significant in zip(
                demo_results['overall_tests'].items(), pvals_corrected, rejected
            ):
                test['corrected_p_value'] = corrected_pval
                test['significant'] = is_significant
                
                if is_significant:
                    criterion_col = f'crit_{criterion}'
                    
                    # Get basic selection stats
                    rates = self.dummy_data.groupby(self.survey_data[demo_var])[criterion_col].mean()
                    counts = self.dummy_data.groupby(self.survey_data[demo_var])[criterion_col].sum()
                    group_sizes = self.dummy_data.groupby(self.survey_data[demo_var])[criterion_col].size()
                    
                    selection_stats = {
                        group: {
                            'rate': rate,
                            'count': int(counts[group]),
                            'total': int(group_sizes[group])
                        }
                        for group, rate in rates.items()
                    }
                    
                    # Get pairwise comparisons
                    pairwise_results = self.pairwise_group_comparisons(demo_var, criterion_col)
                    
                    demo_results['significant_associations'].append({
                        'criterion': criterion,
                        'effect_size': test['effect_size'],
                        'corrected_p_value': corrected_pval,
                        'selection_stats': selection_stats,
                        'pairwise_comparisons': pairwise_results
                    })
            
            demo_criteria_results[demo_var] = demo_results
        
        self.results['demographic_criteria_selection'] = demo_criteria_results

    def analyze_criteria_importance_lacking(self):
        """Analyze discrepancy between importance and lacking criteria."""
        # Get criteria columns (removing the 'crit_' and 'lack_' prefixes for matching)
        criteria_cols = [col.replace('crit_', '') for col in self.dummy_data.columns if col.startswith('crit_')]
        
        # Initialize results
        comparison_results = []
        
        for criterion in criteria_cols:
            # Get importance and lacking percentages
            importance = self.dummy_data[f'crit_{criterion}'].mean() * 100
            lacking = self.dummy_data[f'lack_{criterion}'].mean() * 100
            
            # Calculate difference
            difference = importance - lacking
            
            # Create contingency table for McNemar's test
            table = pd.crosstab(
                self.dummy_data[f'crit_{criterion}'],
                self.dummy_data[f'lack_{criterion}']
            )
            
            # Perform McNemar's test
            try:
                # McNemar's test requires a 2x2 table
                if table.shape == (2, 2):
                    result = mcnemar(table, exact=True)  # Using exact=True for better accuracy
                    pvalue = result.pvalue
                else:
                    # Fallback to chi-square if we don't have a proper 2x2 table
                    _, pvalue = stats.chi2_contingency(table)[:2]
            except Exception as e:
                # If any error occurs during statistical testing, use chi-square as fallback
                _, pvalue = stats.chi2_contingency(table)[:2]
            
            comparison_results.append({
                'criterion': criterion,
                'importance': importance,
                'lacking': lacking,
                'difference': difference,
                'p_value': pvalue
            })
        
        # Sort by absolute difference
        comparison_results.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        # Separate into more important vs more lacking
        more_important = [r for r in comparison_results if r['difference'] > 0]
        more_lacking = [r for r in comparison_results if r['difference'] <= 0]
        
        # Sort each group by difference magnitude
        more_important.sort(key=lambda x: x['difference'], reverse=True)
        more_lacking.sort(key=lambda x: x['difference'])
        
        self.results['criteria_comparison'] = {
            'more_important': more_important,
            'more_lacking': more_lacking,
            'all_results': comparison_results
        }


    def analyze_research_goals(self):
        """Analyze research goal patterns and their relationship with criteria and LLM perception."""
        # Get research goal columns
        goal_cols = [col for col in self.dummy_data.columns if col.startswith('goal_')]
        
        # Calculate frequencies
        frequencies = self.dummy_data[goal_cols].sum() / len(self.dummy_data)
        
        # Calculate correlation matrix between goals
        cooccurrence = np.corrcoef(self.dummy_data[goal_cols].T)
        
        # Calculate correlations with criteria selection
        criteria_cols = [col for col in self.dummy_data.columns if col.startswith('crit_')]
        goal_criteria_correlations = np.zeros((len(goal_cols), len(criteria_cols)))
        
        for i, goal in enumerate(goal_cols):
            for j, criterion in enumerate(criteria_cols):
                contingency = pd.crosstab(
                    self.dummy_data[goal],
                    self.dummy_data[criterion]
                )
                
                # Calculate phi coefficient
                chi2, _ = stats.chi2_contingency(contingency)[:2]
                n = contingency.sum().sum()
                phi = np.sqrt(chi2 / n)
                
                # Adjust sign based on correlation
                if stats.pearsonr(self.dummy_data[goal], self.dummy_data[criterion])[0] < 0:
                    phi = -phi
                
                goal_criteria_correlations[i, j] = phi
        
        # Analyze relationship with LLM perception
        llm_perception_by_goals = {}
        for goal in goal_cols:
            contingency = pd.crosstab(
                self.dummy_data[goal],
                self.survey_data['llm_intelligence']
            )
            
            # Fisher's exact test
            _, pval = stats.fisher_exact(contingency)
            
            # Effect size (Cramer's V)
            chi2, _, _, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim))
            
            # Calculate proportions
            proportions = pd.crosstab(
                self.dummy_data[goal],
                self.survey_data['llm_intelligence'],
                normalize='index'
            )
            
            llm_perception_by_goals[goal] = {
                'fisher_pval': pval,
                'effect_size': cramer_v,
                'perception_breakdown': proportions
            }
        
        self.results['research_goals'] = {
            'frequencies': frequencies,
            'cooccurrence': cooccurrence,
            'goal_criteria_correlations': pd.DataFrame(
                goal_criteria_correlations,
                index=goal_cols,
                columns=criteria_cols
            ),
            'llm_perception': llm_perception_by_goals
        }
        
    def run_statistical_tests(self):
        """Run all statistical tests with multiple comparison corrections."""
        pvals = []
        test_names = []
        
        # Add LLM perception p-values from demographic patterns
        for demo_var, pattern in self.results['llm_perception']['demographic_patterns'].items():
            pvals.append(pattern['fisher_pval'])
            test_names.append(f'llm_perception_{demo_var}')
        
        # Add research goal LLM perception tests
        for goal, stats in self.results['research_goals']['llm_perception'].items():
            pvals.append(stats['fisher_pval'])
            test_names.append(f'llm_perception_goal_{goal}')
        
        # Add LLM perception chi-square p-value
        pvals.append(self.results['llm_perception']['chi2_test']['pvalue'])
        test_names.append('chi2_llm_perception')
        
        # Correct for multiple comparisons
        rejected, pvals_corrected, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method='bonferroni'
        )
        
        self.results['multiple_testing'] = {
            'original_pvals': dict(zip(test_names, pvals)),
            'corrected_pvals': dict(zip(test_names, pvals_corrected)),
            'rejected_null': dict(zip(test_names, rejected))
        }
        
    def generate_summary_report(self):
        """Generate a comprehensive summary of all analyses."""
        report = {
            'criteria_agreement': {
                'most_common_criteria': self.results['criteria_agreement']['frequencies'].nlargest(3),
                'least_common_criteria': self.results['criteria_agreement']['frequencies'].nsmallest(3)
            },
            'llm_perception': {
                'current_perception': self.results['llm_perception']['current_distribution'],
                'future_outlook': self.results['llm_perception']['future_distribution'],
                'chi2_significance': self.results['llm_perception']['chi2_test']['pvalue'] < 0.05,
                'demographic_patterns': {
                    var: {
                        'effect_size': pattern['effect_size'],
                        'significance': self.results['multiple_testing']['rejected_null'][f'llm_perception_{var}']
                    }
                    for var, pattern in self.results['llm_perception']['demographic_patterns'].items()
                }
            },
            'criteria_correlations': {
                'strongest_correlations': self._get_top_correlations(3),
                'weakest_correlations': self._get_top_correlations(3, ascending=True)
            }
        }
        
        self.results['summary_report'] = report
        
    def _get_top_correlations(self, n=3, ascending=False):
        """Helper function to get top correlations from phi matrix."""
        phi_df = self.results['criteria_correlations']['phi_matrix']
        corr_pairs = []
        
        for i in range(len(phi_df)):
            for j in range(i + 1, len(phi_df)):
                corr_pairs.append({
                    'criteria_1': phi_df.index[i],
                    'criteria_2': phi_df.index[j],
                    'correlation': phi_df.iloc[i, j]
                })
                
        sorted_pairs = sorted(
            corr_pairs,
            key=lambda x: x['correlation'],
            reverse=not ascending
        )
        return sorted_pairs[:n]

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        # Save raw data to the analysis class
        self.results['survey_data'] = self.survey_data
        self.results['dummy_data'] = self.dummy_data

        # Run all analyses
        self.analyze_criteria_agreement()
        self.analyze_llm_perception()
        self.analyze_criteria_correlations()
        self.analyze_research_goals()  # Add this line
        self.analyze_demographic_criteria_selection()
        self.analyze_criteria_importance_lacking()
        self.run_statistical_tests()
        self.generate_summary_report()
        
        return self.results