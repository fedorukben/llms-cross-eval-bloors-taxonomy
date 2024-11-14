import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
from collections import defaultdict
from enum import Enum
import textwrap
from matplotlib import font_manager

class DetailedLLMAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with an existing DataFrame from the basic analysis."""
        self.df = df
        self.llm_names = sorted(df['reviewer_llm'].unique())
        
    def analyze_grading_bias(self) -> pd.DataFrame:
        """
        Analyze whether LLMs show systematic bias in grading each other.
        Returns DataFrame with bias statistics.
        """
        # Calculate overall mean grade for reference
        overall_mean = self.df['grade'].mean()
        
        # Create bias matrix
        bias_data = []
        for reviewer in self.llm_names:
            for reviewed in self.llm_names:
                if reviewer != reviewed:  # Skip self-evaluations
                    grades = self.df[
                        (self.df['reviewer_llm'] == reviewer) & 
                        (self.df['reviewed_llm'] == reviewed)
                    ]['grade']
                    
                    if len(grades) > 0:
                        bias = grades.mean() - overall_mean
                        # Perform t-test against overall mean
                        t_stat, p_value = stats.ttest_1samp(grades, overall_mean)
                        
                        bias_data.append({
                            'reviewer': reviewer,
                            'reviewed': reviewed,
                            'bias': round(bias, 2),
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
        
        return pd.DataFrame(bias_data)
    
    def find_controversial_cases(self, std_threshold: float = 10.0) -> pd.DataFrame:
        """
        Identify and analyze cases with high disagreement between reviewers.
        """
        # Group by response and calculate disagreement metrics
        response_stats = self.df.groupby(['reviewed_llm', 'original_response']).agg({
            'grade': ['mean', 'std', 'count', lambda x: x.max() - x.min()]
        }).reset_index()
        
        response_stats.columns = ['reviewed_llm', 'response', 'mean_grade', 'std_dev', 'num_reviews', 'grade_range']
        
        # Filter for high disagreement cases
        controversial = response_stats[response_stats['std_dev'] >= std_threshold].copy()
        controversial = controversial.sort_values('std_dev', ascending=False)
        
        # Add individual grades and reviewers for these cases
        detailed_cases = []
        for _, case in controversial.iterrows():
            case_grades = self.df[
                (self.df['reviewed_llm'] == case['reviewed_llm']) &
                (self.df['original_response'] == case['response'])
            ][['reviewer_llm', 'grade', 'evaluation_text']]
            
            detailed_cases.append({
                'reviewed_llm': case['reviewed_llm'],
                'mean_grade': case['mean_grade'],
                'std_dev': case['std_dev'],
                'grade_range': case['grade_range'],
                'individual_grades': case_grades.to_dict('records')
            })
            
        return pd.DataFrame(detailed_cases)
    
    def analyze_reviewer_consistency(self) -> pd.DataFrame:
        """
        Analyze how consistent each LLM is in their grading patterns.
        """
        consistency_data = []
        
        for reviewer in self.llm_names:
            reviewer_grades = self.df[self.df['reviewer_llm'] == reviewer]['grade']
            
            # Calculate various consistency metrics
            consistency_data.append({
                'reviewer': reviewer,
                'mean_grade': reviewer_grades.mean(),
                'median_grade': reviewer_grades.median(),
                'std_dev': reviewer_grades.std(),
                'grade_range': reviewer_grades.max() - reviewer_grades.min(),
                'skewness': reviewer_grades.skew(),
                'num_reviews': len(reviewer_grades)
            })
        
        return pd.DataFrame(consistency_data)
    
    def plot_grade_distributions(self, save_path: str = None):
        """Create violin plots showing grade distribution patterns."""
        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Function to clean LLM names
        def clean_llm_name(name):
            return name.replace('LLM.', '').title()
        
        # Create copy of data with cleaned names
        df_plot = self.df.copy()
        df_plot['reviewer_llm'] = df_plot['reviewer_llm'].apply(clean_llm_name)
        df_plot['reviewed_llm'] = df_plot['reviewed_llm'].apply(clean_llm_name)
        
        # Plot as reviewer
        plt.subplot(2, 1, 1)
        sns.violinplot(
            data=df_plot, 
            x='reviewer_llm', 
            y='grade',
            color='#4C4C4C',  # Dark gray
            inner='box'       # Show box plot inside violin
        )
        plt.title('Grade Distributions by Reviewer LLM', 
                fontsize=12, 
                pad=15)
        plt.xlabel('Reviewer LLM', 
                fontsize=11, 
                labelpad=10)
        plt.ylabel('Grade', 
                fontsize=11, 
                labelpad=10)
        plt.xticks(rotation=45, ha='right')
        
        # Plot as reviewed
        plt.subplot(2, 1, 2)
        sns.violinplot(
            data=df_plot, 
            x='reviewed_llm', 
            y='grade',
            color='#4C4C4C',  # Dark gray
            inner='box'       # Show box plot inside violin
        )
        plt.title('Grade Distributions by Reviewed LLM', 
                fontsize=12, 
                pad=15)
        plt.xlabel('Reviewed LLM', 
                fontsize=11, 
                labelpad=10)
        plt.ylabel('Grade', 
                fontsize=11, 
                labelpad=10)
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_detailed_report(self) -> str:
        """Generate a comprehensive analysis report."""
        # Get analysis results
        bias_analysis = self.analyze_grading_bias()
        consistency_analysis = self.analyze_reviewer_consistency()
        controversial_cases = self.find_controversial_cases()
        
        # Format the report
        report = [
            "# Detailed LLM Peer Review Analysis",
            
            "\n## 1. Significant Grading Biases",
            "Pairs with statistically significant bias (p < 0.05):",
        ]
        
        # Add significant biases
        significant_biases = bias_analysis[bias_analysis['significant']].sort_values('bias', ascending=False)
        for _, bias in significant_biases.iterrows():
            report.append(
                f"- {bias['reviewer']} → {bias['reviewed']}: "
                f"{bias['bias']:+.2f} points (p={bias['p_value']:.4f})"
            )
        
        # Add reviewer consistency
        report.extend([
            "\n## 2. Reviewer Consistency Analysis",
            "\nMost consistent reviewers (by std dev):"
        ])
        
        consistency_sorted = consistency_analysis.sort_values('std_dev')
        for _, reviewer in consistency_sorted.iterrows():
            report.append(
                f"- {reviewer['reviewer']}: σ={reviewer['std_dev']:.2f}, "
                f"mean={reviewer['mean_grade']:.2f}, "
                f"range={reviewer['grade_range']:.2f}"
            )
        
        # Add controversial cases
        report.extend([
            "\n## 3. Most Controversial Evaluations",
            "\nCases with highest disagreement between reviewers:"
        ])
        
        for _, case in controversial_cases.head().iterrows():
            report.append(f"\nReviewed LLM: {case['reviewed_llm']}")
            report.append(f"Mean Grade: {case['mean_grade']:.2f}")
            report.append(f"Std Dev: {case['std_dev']:.2f}")
            report.append("Individual grades:")
            for grade_info in case['individual_grades']:
                report.append(f"  - {grade_info['reviewer_llm']}: {grade_info['grade']}")
        
        return "\n".join(report)

class ResponsePatternAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the existing DataFrame."""
        self.df = df
        
    def analyze_response_patterns(self, min_std: float = 15.0) -> pd.DataFrame:
        """
        Analyze patterns in responses that cause high disagreement.
        Returns detailed analysis of controversial cases.
        """
        # Group by response and calculate statistics
        response_stats = self.df.groupby(
            ['reviewed_llm', 'original_response']
        ).agg({
            'grade': ['mean', 'std', 'count', lambda x: x.max() - x.min()],
            'evaluation_text': 'first'  # Keep one evaluation text for reference
        }).reset_index()
        
        # Flatten column names
        response_stats.columns = [
            'reviewed_llm', 'response', 'mean_grade', 'std_dev', 
            'num_reviews', 'grade_range', 'example_eval'
        ]
        
        # Filter for controversial cases
        controversial = response_stats[response_stats['std_dev'] >= min_std].copy()
        
        # Add grade distribution details
        controversial['grade_distribution'] = controversial.apply(
            lambda x: self._get_grade_distribution(x['reviewed_llm'], x['response']),
            axis=1
        )
        
        return controversial.sort_values('std_dev', ascending=False)
    
    def _get_grade_distribution(self, llm: str, response: str) -> Dict:
        """Get detailed grade distribution for a specific response."""
        grades = self.df[
            (self.df['reviewed_llm'] == llm) & 
            (self.df['original_response'] == response)
        ][['reviewer_llm', 'grade', 'evaluation_text']]
        
        return grades.to_dict('records')

    def plot_controversy_patterns(self, save_path: str = None):
        """Create visualization of grading patterns in controversial cases."""
        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Function to clean LLM names
        def clean_llm_name(name):
            return name.replace('LLM.', '').title()
        
        # Get controversial cases
        controversial = self.analyze_response_patterns()
        
        # Create subplot for grade distributions
        plt.subplot(2, 1, 1)
        controversial_grades = []
        for _, case in controversial.iterrows():
            reviewed_llm = clean_llm_name(case['reviewed_llm'])
            for grade_info in case['grade_distribution']:
                controversial_grades.append({
                    'Case': f"{reviewed_llm} (σ={case['std_dev']:.1f})",
                    'Grade': grade_info['grade'],
                    'Reviewer': clean_llm_name(grade_info['reviewer_llm'])
                })
        
        df_plot = pd.DataFrame(controversial_grades)
        
        # Plot grade distribution with grayscale palette
        grey_palette = sns.color_palette("Greys", n_colors=len(df_plot['Reviewer'].unique()))
        sns.stripplot(data=df_plot, x='Case', y='Grade', hue='Reviewer',
                    jitter=0.2, size=10, alpha=0.6, palette=grey_palette)
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Grade Distribution in Controversial Cases', 
                fontsize=12, pad=15)
        plt.xlabel('Case', fontsize=11, labelpad=10)
        plt.ylabel('Grade', fontsize=11, labelpad=10)
        
        # Adjust legend
        plt.legend(title='Reviewer', title_fontsize=10, fontsize=9)
        
        # Create subplot for reviewer behavior
        plt.subplot(2, 1, 2)
        reviewer_stats = df_plot.groupby('Reviewer').agg({
            'Grade': ['mean', 'std']
        }).reset_index()
        reviewer_stats.columns = ['Reviewer', 'Mean Grade', 'Std Dev']
        
        x = np.arange(len(reviewer_stats))
        width = 0.35
        
        # Plot bars in grayscale
        plt.bar(x - width/2, reviewer_stats['Mean Grade'], width,
            label='Mean Grade', color='#A0A0A0')
        plt.bar(x + width/2, reviewer_stats['Std Dev'], width,
            label='Std Dev', color='#D3D3D3')
        
        plt.xlabel('Reviewer', fontsize=11, labelpad=10)
        plt.ylabel('Score', fontsize=11, labelpad=10)
        plt.title('Reviewer Behavior in Controversial Cases', 
                fontsize=12, pad=15)
        
        # Clean reviewer names and set ticks
        plt.xticks(x, reviewer_stats['Reviewer'], rotation=45, ha='right')
        
        # Adjust legend
        plt.legend(fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_pairwise_disagreement(self, save_path: str = None):
        """Create heatmap of average disagreement between LLM pairs."""
        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        
        plt.figure(figsize=(12, 8))
        
        # Function to clean LLM names
        def clean_llm_name(name):
            return name.replace('LLM.', '').title()
        
        # Calculate average absolute difference in grades between each pair
        llms = sorted(self.df['reviewer_llm'].unique())
        disagreement_matrix = np.zeros((len(llms), len(llms)))
        
        for i, llm1 in enumerate(llms):
            for j, llm2 in enumerate(llms):
                if llm1 != llm2:
                    # Find common responses graded by both
                    responses = set(self.df[self.df['reviewer_llm'] == llm1]['original_response']) & \
                            set(self.df[self.df['reviewer_llm'] == llm2]['original_response'])
                    
                    differences = []
                    for resp in responses:
                        grade1 = float(self.df[(self.df['reviewer_llm'] == llm1) &
                                    (self.df['original_response'] == resp)]['grade'].iloc[0])
                        grade2 = float(self.df[(self.df['reviewer_llm'] == llm2) &
                                    (self.df['original_response'] == resp)]['grade'].iloc[0])
                        differences.append(abs(grade1 - grade2))
                    
                    if differences:
                        disagreement_matrix[i, j] = np.mean(differences)
        
        # Create mask for diagonal (self-comparisons)
        mask = np.eye(len(llms), dtype=bool)
        
        # Clean LLM names
        clean_llms = [clean_llm_name(llm) for llm in llms]
        
        # Plot heatmap with grayscale colormap
        sns.heatmap(
            disagreement_matrix,
            xticklabels=clean_llms,
            yticklabels=clean_llms,
            annot=True,
            fmt='.1f',
            cmap='Greys',
            mask=mask,
            cbar_kws={'label': 'Average Grade Difference'}
        )
        
        plt.title('Average Grade Disagreement Between LLM Pairs', 
                fontsize=12, 
                pad=15)
        plt.xlabel('LLM 2', 
                fontsize=11, 
                labelpad=10)
        plt.ylabel('LLM 1', 
                fontsize=11, 
                labelpad=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_response_report(self) -> str:
        """Generate detailed report about response patterns."""
        controversial = self.analyze_response_patterns()
        
        report = ["# Response Pattern Analysis\n"]
        
        # Overall statistics
        report.append("## Overall Controversy Statistics")
        report.append(f"Total controversial responses: {len(controversial)}")
        report.append(f"Average std dev in controversial cases: {controversial['std_dev'].mean():.2f}")
        report.append(f"Maximum grade range: {controversial['grade_range'].max():.2f}")
        
        # Analyze patterns in controversial cases
        report.append("\n## Common Patterns in Controversial Cases")
        
        # Count how often each LLM appears in controversial cases
        controversy_counts = controversial['reviewed_llm'].value_counts()
        report.append("\nLLMs most often in controversial cases:")
        for llm, count in controversy_counts.items():
            report.append(f"- {llm}: {count} cases")
        
        # Analyze reviewer behavior in controversial cases
        report.append("\n## Reviewer Behavior in Controversial Cases")
        for _, case in controversial.head(5).iterrows():
            report.append(f"\nCase for {case['reviewed_llm']} (std_dev: {case['std_dev']:.2f}):")
            
            # Sort grades from highest to lowest
            grades = pd.DataFrame(case['grade_distribution']).sort_values('grade', ascending=False)
            for _, grade_info in grades.iterrows():
                report.append(f"- {grade_info['reviewer_llm']}: {grade_info['grade']:.1f}")
                # Add a snippet of the evaluation text
                eval_text = grade_info['evaluation_text']
                if eval_text and len(eval_text) > 100:
                    eval_text = eval_text[:100] + "..."
                report.append(f"  Comment: {eval_text}")
        
        return "\n".join(report)

# Usage example:
def analyze_response_patterns(df):
    analyzer = ResponsePatternAnalyzer(df)
    
    # Generate visualizations
    analyzer.plot_controversy_patterns('controversy_patterns.png')
    analyzer.plot_pairwise_disagreement('pairwise_disagreement.png')
    
    # Print detailed report
    print(analyzer.generate_response_report())
    
    return analyzer

class LLM(Enum):
    CHATGPT = 1
    CLAUDE = 2
    PERPLEXITY = 3
    GEMINI = 4
    MISTRAL = 5
    COHERE = 6
    GROK = 7

# Usage example:
def analyze_detailed(analyzer):
    """Create detailed analysis from existing analyzer."""
    detailed = DetailedLLMAnalyzer(analyzer.df)
    
    # Generate and save visualizations
    detailed.plot_grade_distributions('grade_distributions.png')
    
    # Print detailed report
    print(detailed.generate_detailed_report())
    
    return detailed

class LLMPeerReviewAnalyzer:
    def __init__(self, data: List[Tuple[float, str, str, str, str]]):
        """
        Initialize analyzer with raw data tuples.
        
        Args:
            data: List of tuples (grade, evaluation_text, reviewer_llm, reviewed_llm, original_response)
        """
        # Convert the data tuples to ensure strings for LLM names
        processed_data = []
        for grade, eval_text, reviewer, reviewed, response in data:
            processed_data.append((
                grade,
                str(eval_text),
                str(reviewer),
                str(reviewed),
                str(response)
            ))
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(processed_data, columns=[
            'grade', 'evaluation_text', 'reviewer_llm', 
            'reviewed_llm', 'original_response'
        ])
        
        # Convert grades to float, handling potential string values
        self.df['grade'] = pd.to_numeric(self.df['grade'], errors='coerce')
        
        # Remove any rows where grade conversion failed
        invalid_grades = self.df['grade'].isna().sum()
        if invalid_grades > 0:
            print(f"Warning: {invalid_grades} grades could not be converted to numbers and were removed")
            self.df = self.df.dropna(subset=['grade'])
        
        # Print unique LLM names for verification
        print("\nUnique reviewer LLMs:", sorted(self.df['reviewer_llm'].unique()))
        print("Unique reviewed LLMs:", sorted(self.df['reviewed_llm'].unique()))
        
        # Cache common calculations
        self._compute_basic_statistics()
    
    def _compute_basic_statistics(self):
        """Compute and cache basic statistics about the evaluations."""
        self.overall_stats = {
            'mean_grade': self.df['grade'].mean(),
            'median_grade': self.df['grade'].median(),
            'std_grade': self.df['grade'].std(),
            'total_evaluations': len(self.df)
        }
        
        # Compute average grades given and received by each LLM
        self.llm_stats = {
            'given': self.df.groupby('reviewer_llm')['grade'].agg(['mean', 'std', 'count']).round(2),
            'received': self.df.groupby('reviewed_llm')['grade'].agg(['mean', 'std', 'count']).round(2)
        }
    
    def get_bias_matrix(self) -> pd.DataFrame:
        """
        Create a matrix showing grading bias between LLMs.
        Returns difference from mean grade for each reviewer-reviewed pair.
        """
        # Create pivot table of average grades
        bias_matrix = pd.pivot_table(
            self.df, 
            values='grade',
            index='reviewer_llm',
            columns='reviewed_llm',
            aggfunc='mean'
        )
        
        # Subtract each reviewer's mean grade to show bias
        reviewer_means = self.df.groupby('reviewer_llm')['grade'].mean()
        bias_matrix = bias_matrix.sub(reviewer_means, axis=0)
        
        return bias_matrix.round(2)
    
    def analyze_agreement(self) -> Dict:
        """
        Analyze agreement between different LLMs' evaluations.
        Returns statistics about evaluation consistency.
        """
        # Group by reviewed_llm and original_response to get all grades for same response
        grouped = self.df.groupby(['reviewed_llm', 'original_response'])
        
        agreement_stats = {
            'mean_std': grouped['grade'].std().mean(),
            'max_std': grouped['grade'].std().max(),
            'grade_range': grouped['grade'].agg(lambda x: x.max() - x.min()).mean()
        }
        
        return {k: round(v, 2) for k, v in agreement_stats.items()}
    
    def find_controversial_evaluations(self, std_threshold: float = 15.0) -> pd.DataFrame:
        """
        Find evaluations with high disagreement between reviewers.
        
        Args:
            std_threshold: Standard deviation threshold for considering an evaluation controversial
        """
        grouped = self.df.groupby(['reviewed_llm', 'original_response'])
        std_scores = grouped['grade'].std()
        controversial = std_scores[std_scores > std_threshold]
        
        # Get full details for controversial evaluations
        controversial_evals = self.df[
            self.df.set_index(['reviewed_llm', 'original_response'])
                .index
                .isin(controversial.index)
        ]
        
        return controversial_evals
    
    def plot_grade_distribution(self, save_path: str = None):
        """Generate visualization of grade distributions by LLM."""
        plt.figure(figsize=(15, 6))
        
        # Plot grade distribution for grades given
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x='reviewer_llm', y='grade')
        plt.title('Grade Distribution by Reviewer LLM')
        plt.xticks(rotation=45, ha='right')
        
        # Plot grade distribution for grades received
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='reviewed_llm', y='grade')
        plt.title('Grade Distribution by Reviewed LLM')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_heatmap(self, save_path: str = None):
        """Generate heatmap of average grades between LLMs."""
        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        
        plt.figure(figsize=(10, 8))
        
        # Create matrix of average grades
        grade_matrix = pd.pivot_table(
            self.df,
            values='grade',
            index='reviewer_llm',
            columns='reviewed_llm',
            aggfunc='mean'
        )
        
        # Function to clean LLM names
        def clean_llm_name(name):
            return name.replace('LLM.', '').title()
        
        # Clean up index and column names
        grade_matrix.index = [clean_llm_name(name) for name in grade_matrix.index]
        grade_matrix.columns = [clean_llm_name(name) for name in grade_matrix.columns]
        
        # Plot heatmap with grayscale colormap
        sns.heatmap(
            grade_matrix, 
            annot=True, 
            fmt='.1f', 
            cmap='Greys',
            center=75,
            annot_kws={'fontsize': 10, 'family': 'Times New Roman'},
            cbar_kws={'label': 'Average Grade'}
        )
        
        # Customize text elements with Times New Roman
        plt.title('Average Grades: Reviewer vs Reviewed LLM', 
                fontsize=12, 
                fontfamily='Times New Roman',
                pad=15)
        plt.xlabel('LLM Reviewed', 
                fontsize=11, 
                fontfamily='Times New Roman',
                labelpad=10)
        plt.ylabel('LLM Reviewer', 
                fontsize=11, 
                fontfamily='Times New Roman',
                labelpad=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of the analysis."""
        report = [
            "# LLM Peer Review Analysis Summary",
            "\n## Overall Statistics",
            f"Total Evaluations: {self.overall_stats['total_evaluations']:,}",
            f"Mean Grade: {self.overall_stats['mean_grade']:.2f}",
            f"Median Grade: {self.overall_stats['median_grade']:.2f}",
            f"Standard Deviation: {self.overall_stats['std_grade']:.2f}",
            
            "\n## Reviewer Statistics (grades given)",
            self.llm_stats['given'].to_string(),
            
            "\n## Reviewed LLM Statistics (grades received)",
            self.llm_stats['received'].to_string(),
            
            "\n## Agreement Analysis",
            f"Average Standard Deviation between reviewers: {self.analyze_agreement()['mean_std']:.2f}",
            f"Maximum Standard Deviation: {self.analyze_agreement()['max_std']:.2f}",
            f"Average Grade Range: {self.analyze_agreement()['grade_range']:.2f}"
        ]
        
        return "\n".join(report)

def create_analyzer(data_path: str) -> LLMPeerReviewAnalyzer:
    """
    Helper function to load data from pickle file and create analyzer instance.
    """
    import pickle
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return LLMPeerReviewAnalyzer(data)

analyzer = create_analyzer('crossevals.pkl')
print(analyzer.generate_summary_report())
bias_matrix = analyzer.get_bias_matrix()
controversial = analyzer.find_controversial_evaluations(std_threshold=15.0)
analyzer.plot_grade_distribution('grades.png')
analyzer.plot_heatmap('heatmap.png')
detailed = analyze_detailed(analyzer)
pattern_analyzer = analyze_response_patterns(detailed.df)