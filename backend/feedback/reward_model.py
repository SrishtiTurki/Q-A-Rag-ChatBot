# feedback/reward_model.py

import numpy as np
import json
import os
import pickle
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Setup logging
log = logging.getLogger("rag")

class StarRewardModel:
    """
    Reward model that learns from 1-5 star ratings.
    """
    
    def __init__(self, data_dir: str = "feedback_data"):
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "star_reward_model.pkl")
        self.log_path = os.path.join(data_dir, "feedback_log.json")
        
        # Data structures
        self.chunk_scores = {}
        self.source_weights = {}
        self.chunk_feedback_count = {}
        self.feedback_history = []
        
        # Hyperparameters
        self.learning_rate = 0.3
        self.min_feedback_threshold = 3
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        print(f"[RewardModel] Data directory: {data_dir}")
        
        # Load existing data
        self.load_data()
        print(f"[RewardModel] Initialized with {len(self.chunk_scores)} chunks tracked")
        print(f"[RewardModel] Total feedback entries: {len(self.feedback_history)}")
    
    def load_data(self):
        """Load learned weights from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunk_scores = data.get("chunk_scores", {})
                    self.source_weights = data.get("source_weights", {})
                    self.chunk_feedback_count = data.get("chunk_feedback_count", {})
                    self.feedback_history = data.get("feedback_history", [])
                    print(f"[RewardModel] Loaded: {len(self.chunk_scores)} chunks with feedback")
                    print(f"[RewardModel] Loaded: {len(self.feedback_history)} feedback entries")
            except Exception as e:
                print(f"[RewardModel] Failed to load: {e}")
        else:
            print(f"[RewardModel] No existing data found at: {self.model_path}")
    
    def save_data(self):
        """Save learned weights to disk."""
        try:
            # Save the main model
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "chunk_scores": self.chunk_scores,
                    "source_weights": self.source_weights,
                    "chunk_feedback_count": self.chunk_feedback_count,
                    "feedback_history": self.feedback_history[-1000:]  # Keep last 1000
                }, f)
            
            # Also save as JSON for easy inspection
            with open(self.log_path, "w") as f:
                json.dump(self.feedback_history[-100:], f, indent=2)
            
            print(f"[RewardModel] Saved successfully to: {self.model_path}")
            print(f"[RewardModel] Total feedback saved: {len(self.feedback_history)}")
        except Exception as e:
            print(f"[RewardModel] Failed to save: {e}")
    
    def stars_to_reward(self, stars: int) -> float:
        """Convert star rating to reward signal."""
        return (stars - 3) / 2
    
    def add_feedback(self, question: str, answer: str, chunks: List[Dict], 
                     stars: int, comment: str = "", response_time: float = None):
        """Process feedback with star rating."""
        print(f"[RewardModel] Adding feedback: {stars} stars, {len(chunks)} chunks")
        
        if not chunks:
            print("[RewardModel] WARNING: No chunks to update!")
            return
        
        reward = self.stars_to_reward(stars)
        print(f"[RewardModel] Reward signal: {reward:.2f}")
        
        # Update each chunk used in the answer
        for i, chunk in enumerate(chunks):
            uid = chunk.get("uid", f"{chunk['source']}_{chunk['chunk_index']}")
            
            # Position weight - earlier chunks matter more
            position_weight = 1.0 - (i / len(chunks)) * 0.3
            
            # Update chunk score with moving average
            if uid in self.chunk_scores:
                current = self.chunk_scores[uid]
                count = self.chunk_feedback_count.get(uid, 0)
                alpha = min(0.5, 1.0 / (count + 1))
                self.chunk_scores[uid] = current * (1 - alpha) + reward * position_weight * alpha
                self.chunk_feedback_count[uid] = count + 1
                print(f"[RewardModel] Updated chunk {uid[:30]}: {current:.3f} -> {self.chunk_scores[uid]:.3f}")
            else:
                self.chunk_scores[uid] = reward * position_weight * 0.5
                self.chunk_feedback_count[uid] = 1
                print(f"[RewardModel] New chunk {uid[:30]}: {self.chunk_scores[uid]:.3f}")
            
            # Update source weight
            source = chunk['source'].replace(' [images]', '')
            if source in self.source_weights:
                count = self.source_weights.get(f"{source}_count", 0)
                alpha = min(0.3, 1.0 / (count + 1))
                self.source_weights[source] = self.source_weights.get(source, 0) * (1 - alpha) + reward * alpha
                self.source_weights[f"{source}_count"] = count + 1
            else:
                self.source_weights[source] = reward * 0.3
                self.source_weights[f"{source}_count"] = 1
        
        # Store feedback for analysis
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question[:200],
            "answer": answer[:200],
            "stars": stars,
            "comment": comment,
            "response_time": response_time,
            "chunks_used": [c.get("uid", "") for c in chunks[:5]]
        }
        
        self.feedback_history.append(feedback_entry)
        print(f"[RewardModel] Feedback stored. Total: {len(self.feedback_history)}")
        
        # Save immediately
        self.save_data()
        print(f"[RewardModel] Feedback saved successfully!")
    
    def rerank_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks using learned reward signals."""
        scored_chunks = []
        
        for chunk in chunks:
            uid = chunk.get("uid", f"{chunk['source']}_{chunk['chunk_index']}")
            source = chunk['source'].replace(' [images]', '')
            
            base_score = chunk.get("score", 0.0)
            chunk_reward = self.chunk_scores.get(uid, 0.0)
            chunk_feedback_count = self.chunk_feedback_count.get(uid, 0)
            source_weight = self.source_weights.get(source, 0.0)
            
            if chunk_feedback_count >= self.min_feedback_threshold:
                adjusted_score = (
                    base_score * 0.6 +
                    chunk_reward * 0.25 +
                    source_weight * 0.15
                )
            else:
                adjusted_score = base_score * 0.8 + source_weight * 0.2
            
            scored_chunks.append({
                **chunk,
                "original_score": base_score,
                "reward_score": chunk_reward,
                "source_weight": source_weight,
                "feedback_count": chunk_feedback_count,
                "adjusted_score": adjusted_score
            })
        
        scored_chunks.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return scored_chunks
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        chunks_with_feedback = len([c for c in self.chunk_scores if self.chunk_feedback_count.get(c, 0) >= 3])
        
        # Distribution of feedback
        stars_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for f in self.feedback_history:
            stars = f.get("stars", 3)
            if stars in stars_dist:
                stars_dist[stars] += 1
        
        total_feedback = sum(stars_dist.values())
        
        print(f"[RewardModel] Stats: total={total_feedback}, avg={sum(f.get('stars',3) for f in self.feedback_history)/total_feedback if total_feedback>0 else 0}")
        
        return {
            "total_feedback": total_feedback,
            "chunks_with_feedback": chunks_with_feedback,
            "total_chunks_tracked": len(self.chunk_scores),
            "sources_tracked": len([s for s in self.source_weights if not s.endswith("_count")]),
            "stars_distribution": stars_dist,
            "average_rating": (
                sum(f.get("stars", 3) for f in self.feedback_history) / total_feedback
                if total_feedback > 0 else 0
            )
        }
    
    def get_best_sources(self, top_k: int = 5) -> List[Dict]:
        """Get top performing sources."""
        sources = []
        for key, value in self.source_weights.items():
            if not key.endswith("_count"):
                count_key = f"{key}_count"
                count = self.source_weights.get(count_key, 0)
                if count >= 2:
                    sources.append({
                        "source": key,
                        "score": value,
                        "feedback_count": count
                    })
        
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources[:top_k]
    
    def get_poor_sources(self, top_k: int = 5) -> List[Dict]:
        """Get worst performing sources."""
        sources = []
        for key, value in self.source_weights.items():
            if not key.endswith("_count"):
                count_key = f"{key}_count"
                count = self.source_weights.get(count_key, 0)
                if count >= 2:
                    sources.append({
                        "source": key,
                        "score": value,
                        "feedback_count": count
                    })
        
        sources.sort(key=lambda x: x["score"])
        return sources[:top_k]

    def reset(self):
        """Reset all learned data."""
        self.chunk_scores = {}
        self.source_weights = {}
        self.chunk_feedback_count = {}
        self.feedback_history = []
        self.save_data()
        print("[RewardModel] Reset all data")


# Global instance
reward_model = StarRewardModel()