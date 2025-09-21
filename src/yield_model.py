"""
Debone Yield Prediction Model
Predicts meat yield % from bird characteristics using Random Forest
Built for Twin Rivers Foods deboning optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class DeboneYieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_columns = ['bird_weight_lbs', 'bird_age_days', 'feed_quality_score']
        self.is_trained = False
        os.makedirs('models', exist_ok=True)
    
    def prepare_features(self, df):
        """Extract and engineer features for prediction"""
        X = df[self.feature_columns].copy()
        # Add interaction term for better predictions
        X['weight_age_interaction'] = X['bird_weight_lbs'] * X['bird_age_days']
        return X
    
    def train(self, data_path):
        """Train model on historical deboning data"""
        print(f"🤖 Training model on {data_path}...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"📊 Loaded {len(df)} samples")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['actual_yield_pct']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate performance
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"✅ Training Complete!")
        print(f"📈 Training R²: {train_r2:.3f}")
        print(f"📈 Test R²: {test_r2:.3f}")
        print(f"🎯 Test MAE: {test_mae:.2f}%")
        
        # Save model for production use
        joblib.dump(self.model, 'models/yield_predictor.pkl')
        print("💾 Model saved to models/yield_predictor.pkl")
        
        return self
    
    def predict(self, new_data):
        """Predict yield for new birds"""
        if not self.is_trained:
            raise ValueError("❌ Model must be trained first! Call .train()")
        
        if isinstance(new_data, pd.DataFrame):
            X = self.prepare_features(new_data)
        else:
            # Single bird prediction
            X = pd.DataFrame([new_data], columns=self.feature_columns)
            X['weight_age_interaction'] = X['bird_weight_lbs'] * X['bird_age_days']
        
        predictions = self.model.predict(X)
        return predictions
    
    def feature_importance(self):
        """Show which factors drive yield predictions"""
        importances = pd.DataFrame({
            'feature': self.feature_columns + ['weight_age_interaction'],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importances
    
    def quick_demo(self):
        """Run a quick demo prediction"""
        if not self.is_trained:
            print("❌ Train model first!")
            return
        
        # Sample bird for demo
        sample_bird = {
            'bird_weight_lbs': 6.2,
            'bird_age_days': 48,
            'feed_quality_score': 0.92
        }
        
        pred_yield = self.predict(sample_bird)[0]
        meat_weight = sample_bird['bird_weight_lbs'] * (pred_yield / 100)
        
        print(f"\n🎯 SAMPLE PREDICTION:")
        print(f"Bird: {sample_bird['bird_weight_lbs']:.1f}lbs, {sample_bird['bird_age_days']} days")
        print(f"Feed Quality: {sample_bird['feed_quality_score']:.2f}")
        print(f"Predicted Yield: {pred_yield:.1f}%")
        print(f"Expected Meat: {meat_weight:.2f}lbs")
        
        # ROI calculation
        value_per_lb = 1.89  # Wholesale price
        total_value = meat_weight * value_per_lb
        print(f"💰 Revenue Potential: ${total_value:.2f} per bird")

# Quick test function
if __name__ == "__main__":
    print("🧪 Running model test...")
    predictor = DeboneYieldPredictor()
    predictor.train('data/synthetic_debone_data.csv')
    predictor.quick_demo()
    print("\n✅ Model test complete!")
