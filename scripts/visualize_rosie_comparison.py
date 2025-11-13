#!/usr/bin/env python3
"""
Create a visual comparison of ROSIE vs IF2RNA pipelines
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('ROSIE + IF2RNA: Complementary Technologies', fontsize=16, fontweight='bold')

# Color scheme
color_he = '#FF6B6B'      # H&E
color_if = '#4ECDC4'      # IF
color_gene = '#45B7D1'    # Gene expression
color_rosie = '#95E1D3'   # ROSIE
color_if2rna = '#F38181'  # IF2RNA

def draw_box(ax, x, y, width, height, text, color, alpha=0.8):
    """Draw a colored box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color, 
                         edgecolor='black',
                         linewidth=2,
                         alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', 
           fontsize=11, fontweight='bold',
           wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=30, 
                           linewidth=3,
                           color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, label, 
               ha='center', va='bottom',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ========= AXIS 1: ROSIE Pipeline =========
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)
ax1.axis('off')
ax1.set_title('ROSIE: H&E → Immunofluorescence Prediction', fontsize=13, fontweight='bold', pad=20)

# H&E input
draw_box(ax1, 0.5, 0.7, 1.5, 0.6, 'H&E\nImage', color_he)
draw_arrow(ax1, 2.0, 1.0, 3.0, 1.0, 'ConvNext\nCNN')

# ROSIE model
draw_box(ax1, 3.0, 0.5, 2.0, 1.0, 'ROSIE\nModel\n(50M params)', color_rosie)
draw_arrow(ax1, 5.0, 1.0, 6.0, 1.0, '50 markers')

# IF output
draw_box(ax1, 6.0, 0.7, 1.5, 0.6, 'mIF\nImage', color_if)

# Stats box
ax1.text(8.5, 1.5, 'Training Data:\n1,342 samples\n16M cells\nPearson R: 0.285', 
        ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ========= AXIS 2: IF2RNA Pipeline =========
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 2)
ax2.axis('off')
ax2.set_title('IF2RNA: Immunofluorescence → Gene Expression Prediction', fontsize=13, fontweight='bold', pad=20)

# IF input
draw_box(ax2, 0.5, 0.7, 1.5, 0.6, 'Multi-channel\nIF Image', color_if)
draw_arrow(ax2, 2.0, 1.0, 3.0, 1.0, 'ResNet50 +\nAttention')

# IF2RNA model  
draw_box(ax2, 3.0, 0.5, 2.0, 1.0, 'IF2RNA\nModel\n(HE2RNA-based)', color_if2rna)
draw_arrow(ax2, 5.0, 1.0, 6.0, 1.0, '18K genes')

# Gene expression output
draw_box(ax2, 6.0, 0.7, 1.5, 0.6, 'Gene\nExpression', color_gene)

# Stats box
ax2.text(8.5, 1.5, 'Training Data:\nGeoMx DSP\n200+ ROIs\n(Your project)', 
        ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ========= AXIS 3: Combined Pipeline =========
ax3 = axes[2]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 2)
ax3.axis('off')
ax3.set_title('COMBINED: H&E → IF → Gene Expression (Novel Pipeline!)', 
             fontsize=13, fontweight='bold', pad=20, color='darkgreen')

# H&E input
draw_box(ax3, 0.3, 0.7, 1.2, 0.6, 'H&E\nImage', color_he)
draw_arrow(ax3, 1.5, 1.0, 2.3, 1.0, 'ROSIE')

# IF intermediate
draw_box(ax3, 2.3, 0.7, 1.2, 0.6, 'IF\nImage', color_if)
draw_arrow(ax3, 3.5, 1.0, 4.3, 1.0, 'IF2RNA')

# Gene expression output
draw_box(ax3, 4.3, 0.7, 1.2, 0.6, 'Gene\nExpression', color_gene)

# Benefits box
benefits_text = """
✓ Uses only H&E (cheap, universal)
✓ Predicts full spatial transcriptome
✓ Novel end-to-end approach
✓ Cost: $100 vs $5,000
"""
ax3.text(6.5, 1.0, benefits_text, 
        ha='left', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.5))

plt.tight_layout()
plt.savefig('/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/docs/ROSIE_IF2RNA_Comparison.png', 
           dpi=300, bbox_inches='tight')
print("✅ Diagram saved to: docs/ROSIE_IF2RNA_Comparison.png")
plt.show()
