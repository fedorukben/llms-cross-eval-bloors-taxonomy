import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
from enum import Enum

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict

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
        plt.figure(figsize=(10, 8))
        
        # Create matrix of average grades
        grade_matrix = pd.pivot_table(
            self.df, 
            values='grade',
            index='reviewer_llm',
            columns='reviewed_llm',
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(grade_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', center=75)
        plt.title('Average Grades: Reviewer vs Reviewed LLM')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
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



class LLM(Enum):
    CHATGPT = 1
    CLAUDE = 2
    PERPLEXITY = 3
    GEMINI = 4
    MISTRAL = 5
    COHERE = 6
    GROK = 7

analyzer = create_analyzer('crossevals.pkl')
print(analyzer.generate_summary_report())
bias_matrix = analyzer.get_bias_matrix()
controversial = analyzer.find_controversial_evaluations(std_threshold=15.0)
analyzer.plot_grade_distribution('grades.png')
analyzer.plot_heatmap('heatmap.png')