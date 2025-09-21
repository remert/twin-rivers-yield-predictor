"""
Generate Synthetic Deboning Data for Twin Rivers Demo
Creates realistic poultry processing data for ML training
"""

import pandas as pd
import numpy as np

# Ensure data directory exists
import os
os.makedirs('data', exist_ok=True)

print("🚀 Generating synthetic deboning data...")
np.random.seed(42)

# Generate 2000 samples of realistic deboning data
n_samples = 2000

data = {
    'batch_id': range(1, n_samples + 1),
    'bird_weight_lbs': np.random.normal(5.8, 0.9, n_samples),  # 5.8 lbs avg broiler
    'bird_age_days': np.random.normal(47, 3, n_samples),        # 47 days avg
    'feed_quality_score': np.random.uniform(0.8, 1.0, n_samples), # 0.8-1.0 scale
    'processing_line': np.random.choice(['Line A', 'Line B', 'Line C'], n_samples),
    'actual_yield_pct': 0,  # Will calculate
    'bone_weight_lbs': 0,   # Will calculate
    'meat_weight_lbs': 0    # Will calculate
}

df = pd.DataFrame(data)

# Calculate realistic yields (70-80% typical for broilers)
df['actual_yield_pct'] = (
    72 + 0.8 * (df['bird_weight_lbs'] - 5.8) + 
    0.4 * (df['bird_age_days'] - 47) + 
    5 * (df['feed_quality_score'] - 0.9) + 
    np.random.normal(0, 2, n_samples)
).clip(65, 85)  # 65-85% realistic range

df['bone_weight_lbs'] = df['bird_weight_lbs'] * (1 - df['actual_yield_pct']/100)
df['meat_weight_lbs'] = df['bird_weight_lbs'] * (df['actual_yield_pct']/100)

# Save to CSV
output_path = 'data/synthetic_debone_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Generated {len(df)} rows of synthetic deboning data")
print(f"💾 Saved to: {output_path}")

# Preview results
print("\n📊 Sample Data:")
print(df[['bird_weight_lbs', 'actual_yield_pct', 'meat_weight_lbs']].head().round(2))

# Data quality summary
print(f"\n📈 Data Summary:")
summary = df[['bird_weight_lbs', 'actual_yield_pct', 'meat_weight_lbs']].describe().round(2)
print(summary)

print(f"\n🎯 Data Quality Check:")
print(f"• Avg Bird Weight: {df['bird_weight_lbs'].mean():.1f} lbs")
print(f"• Avg Yield: {df['actual_yield_pct'].mean():.1f}%")
print(f"• Total Meat Output: {df['meat_weight_lbs'].sum():,.0f} lbs")
print(f"• Yield Range: {df['actual_yield_pct'].min():.0f}% - {df['actual_yield_pct'].max():.0f}%")
