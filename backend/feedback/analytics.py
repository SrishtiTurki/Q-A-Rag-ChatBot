"""
Analytics helpers for the feedback system.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class FeedbackAnalytics:
    """Analyze feedback data and generate insights."""
    
    def __init__(self, data_dir: str = "feedback_data"):
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "feedback_log.json")
        self.model_file = os.path.join(data_dir, "star_reward_model.pkl")
        self.load_data()
    
    def load_data(self):
        """Load feedback data."""
        self.feedback_data = []
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    self.feedback_data = json.load(f)
            except:
                self.feedback_data = []
    
    def get_daily_stats(self, days: int = 7) -> Dict:
        """Get statistics for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [
            f for f in self.feedback_data 
            if datetime.fromisoformat(f.get("timestamp", "2000-01-01")) > cutoff
        ]
        
        total = len(recent)
        if total == 0:
            return {"total": 0, "avg_rating": 0, "distribution": {}}
        
        stars = [f.get("stars", 3) for f in recent]
        
        return {
            "total": total,
            "avg_rating": sum(stars) / total,
            "distribution": {
                1: stars.count(1),
                2: stars.count(2),
                3: stars.count(3),
                4: stars.count(4),
                5: stars.count(5)
            }
        }
    
    def get_common_issues(self, min_stars: int = 3) -> List[str]:
        """Extract common issues from comments with low ratings."""
        issues = []
        for f in self.feedback_data:
            if f.get("stars", 5) <= min_stars:
                comment = f.get("comment", "").lower()
                if comment:
                    # Simple keyword extraction
                    keywords = ["missing", "wrong", "incorrect", "citation", 
                               "source", "irrelevant", "not found", "unclear"]
                    for kw in keywords:
                        if kw in comment:
                            issues.append(kw)
        
        # Count occurrences
        from collections import Counter
        counter = Counter(issues)
        return [f"{k}: {v}" for k, v in counter.most_common(5)]