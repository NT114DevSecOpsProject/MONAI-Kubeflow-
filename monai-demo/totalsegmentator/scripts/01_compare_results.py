"""
Compare MONAI Model Performance on Task09_Spleen vs TotalSegmentator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 100)
print("COMPARE: Task09_Spleen vs TotalSegmentator Performance")
print("=" * 100)

# Load results (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
task09_csv = PROJECT_ROOT / "task09_spleen/outputs/results.csv"
totalseg_csv = SCRIPT_DIR.parent / "outputs/results.csv"

task09_df = pd.read_csv(task09_csv)
totalseg_df = pd.read_csv(totalseg_csv)

print(f"\nTask09_Spleen:     {len(task09_df)} cases")
print(f"TotalSegmentator:  {len(totalseg_df)} cases")

# Extract valid scores
task09_dice = task09_df[task09_df['dice'] >= 0]['dice'].values
task09_iou = task09_df[task09_df['iou'] >= 0]['iou'].values

totalseg_dice = totalseg_df[totalseg_df['dice'] >= 0]['dice'].values
totalseg_iou = totalseg_df[totalseg_df['iou'] >= 0]['iou'].values

# Create comparison visualization
fig = plt.figure(figsize=(16, 10))
fig.suptitle('MONAI Spleen Segmentation Model: Task09_Spleen vs TotalSegmentator\nDomain Shift Impact',
             fontsize=14, fontweight='bold', y=0.995)

# 1. Dice distribution
ax = plt.subplot(2, 3, 1)
x_pos = np.arange(2)
means = [np.mean(task09_dice), np.mean(totalseg_dice)]
stds = [np.std(task09_dice), np.std(totalseg_dice)]
colors = ['green', 'red']
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('Dice Score', fontsize=11, fontweight='bold')
ax.set_title('Dice Score Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Task09_Spleen\n(Training)', 'TotalSegmentator\n(Test)'], fontsize=10)
ax.set_ylim([0, 1])
ax.grid(alpha=0.3, axis='y')
ax.axhline(0.7, color='orange', linestyle='--', linewidth=2, label='Clinical Threshold')
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.05, f'{mean:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.legend()

# 2. IoU distribution
ax = plt.subplot(2, 3, 2)
means = [np.mean(task09_iou), np.mean(totalseg_iou)]
stds = [np.std(task09_iou), np.std(totalseg_iou)]
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('IoU Score', fontsize=11, fontweight='bold')
ax.set_title('IoU Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Task09_Spleen\n(Training)', 'TotalSegmentator\n(Test)'], fontsize=10)
ax.set_ylim([0, 1])
ax.grid(alpha=0.3, axis='y')
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.05, f'{mean:.3f}', ha='center', fontsize=11, fontweight='bold')

# 3. Variability (Coefficient of Variation)
ax = plt.subplot(2, 3, 3)
task09_cv = np.std(task09_dice) / np.mean(task09_dice) * 100
totalseg_cv = np.std(totalseg_dice) / np.mean(totalseg_dice) * 100
cvs = [task09_cv, totalseg_cv]
bars = ax.bar(x_pos, cvs, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
ax.set_title('Variability/Consistency', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Task09_Spleen\n(Training)', 'TotalSegmentator\n(Test)'], fontsize=10)
ax.grid(alpha=0.3, axis='y')
for i, cv in enumerate(cvs):
    ax.text(i, cv + 2, f'{cv:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(20, color='orange', linestyle='--', linewidth=2, label='Low Variability Threshold', alpha=0.7)
ax.legend()

# 4. Dice distribution histogram
ax = plt.subplot(2, 3, 4)
ax.hist(task09_dice, bins=8, alpha=0.6, label='Task09_Spleen', color='green', edgecolor='black')
ax.hist(totalseg_dice, bins=8, alpha=0.6, label='TotalSegmentator', color='red', edgecolor='black')
ax.axvline(np.mean(task09_dice), color='green', linestyle='--', linewidth=2.5, label=f'T09 Mean: {np.mean(task09_dice):.3f}')
ax.axvline(np.mean(totalseg_dice), color='red', linestyle='--', linewidth=2.5, label=f'TS Mean: {np.mean(totalseg_dice):.3f}')
ax.axvline(0.7, color='orange', linestyle=':', linewidth=2, label='Clinical Threshold: 0.7')
ax.set_xlabel('Dice Score', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Dice Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# 5. Performance categories
ax = plt.subplot(2, 3, 5)

def categorize(scores):
    excellent = sum(s > 0.8 for s in scores)
    good = sum((s >= 0.6) & (s <= 0.8) for s in scores)
    fair = sum((s >= 0.4) & (s < 0.6) for s in scores)
    poor = sum(s < 0.4 for s in scores)
    return [excellent, good, fair, poor]

task09_cat = categorize(task09_dice)
totalseg_cat = categorize(totalseg_dice)

categories = ['Excellent\n(>0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.4-0.6)', 'Poor\n(<0.4)']
x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, task09_cat, width, label='Task09_Spleen', color='green', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, totalseg_cat, width, label='TotalSegmentator', color='red', alpha=0.7, edgecolor='black')

ax.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
ax.set_title('Performance Categories', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. Statistics table
ax = plt.subplot(2, 3, 6)
ax.axis('off')

stats_text = f"""
QUANTITATIVE COMPARISON

Task09_Spleen (Training Dataset):
  Dice Mean:        {np.mean(task09_dice):.4f}
  Dice Std:         {np.std(task09_dice):.4f}
  Dice Range:       [{np.min(task09_dice):.4f}, {np.max(task09_dice):.4f}]
  IoU Mean:         {np.mean(task09_iou):.4f}
  Variability:      {task09_cv:.1f}%
  Excellent cases:  {task09_cat[0]}/{len(task09_dice)} ({task09_cat[0]/len(task09_dice)*100:.0f}%)

TotalSegmentator (Test Dataset):
  Dice Mean:        {np.mean(totalseg_dice):.4f}
  Dice Std:         {np.std(totalseg_dice):.4f}
  Dice Range:       [{np.min(totalseg_dice):.4f}, {np.max(totalseg_dice):.4f}]
  IoU Mean:         {np.mean(totalseg_iou):.4f}
  Variability:      {totalseg_cv:.1f}%
  Excellent cases:  {totalseg_cat[0]}/{len(totalseg_dice)} ({totalseg_cat[0]/len(totalseg_dice)*100:.0f}%)

PERFORMANCE DROP:
  Dice:             {(np.mean(task09_dice) - np.mean(totalseg_dice)):.4f} ({(np.mean(task09_dice) - np.mean(totalseg_dice))/np.mean(task09_dice)*100:.1f}%)
  Variability ↑:    {(totalseg_cv - task09_cv):.1f}% increase
"""

ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparison_path = OUTPUT_DIR / "comparison_task09_vs_totalseg.png"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved comparison: {comparison_path}")
plt.close()

# Print summary
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"""
TASK09_SPLEEN (Original Training Dataset):
  ✓ Dice:  {np.mean(task09_dice):.4f} ± {np.std(task09_dice):.4f}  [Excellent!]
  ✓ IoU:   {np.mean(task09_iou):.4f} ± {np.std(task09_iou):.4f}
  ✓ Variability: {task09_cv:.1f}% (Low - Consistent)
  ✓ Excellent cases: {task09_cat[0]}/{len(task09_dice)} ({task09_cat[0]/len(task09_dice)*100:.0f}%)

TOTALSEGMENTATOR (Different Domain):
  ✗ Dice:  {np.mean(totalseg_dice):.4f} ± {np.std(totalseg_dice):.4f}  [Moderate]
  ✗ IoU:   {np.mean(totalseg_iou):.4f} ± {np.std(totalseg_iou):.4f}
  ✗ Variability: {totalseg_cv:.1f}% (High - Inconsistent)
  ✗ Excellent cases: {totalseg_cat[0]}/{len(totalseg_dice)} ({totalseg_cat[0]/len(totalseg_dice)*100:.0f}%)

DOMAIN SHIFT IMPACT:
  ↓ Dice Performance:  -{(np.mean(task09_dice) - np.mean(totalseg_dice)):.4f} ({(np.mean(task09_dice) - np.mean(totalseg_dice))/np.mean(task09_dice)*100:.1f}% drop)
  ↑ Variability:       +{(totalseg_cv - task09_cv):.1f}% increase
  ↓ Consistency:       {task09_cat[0]}/{len(task09_dice)} → {totalseg_cat[0]}/{len(totalseg_dice)} excellent cases

KEY FINDING:
  Model performs MUCH BETTER on its training dataset (Task09_Spleen)!
  Domain shift significantly impacts performance on different data (TotalSegmentator).
  This shows the importance of fine-tuning or domain adaptation.
""")

print("=" * 100)
