"""Generate visualizations for the SFT blog post."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'diagrams')
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
    'savefig.pad_inches': 0.3,
})

MASK_COLOR = '#E8E8E8'    # light gray for masked
TRAIN_COLOR = '#4A90D9'   # blue for trained
PAD_COLOR = '#F5F5F5'     # very light gray for padding
ACCENT = '#D94A4A'        # red for emphasis


# ─────────────────────────────────────────────────────────
# 1. Loss Mask Heatmap
# ─────────────────────────────────────────────────────────
def generate_loss_mask_heatmap():
    """Show which tokens are masked vs trained in one SFT example."""
    # Tokens for: <|user|>Write a greeting<|end|><|assistant|>Good morrow!<|end|>
    tokens =    ['<|user|>', 'W','r','i','t','e',' ','a',' ',
                 'g','r','e','e','t','i','n','g',
                 '<|end|>', '<|asst|>',
                 'G','o','o','d',' ','m','o','r','r','o','w','!',
                 '<|end|>']
    # 0 = masked (user + special), 1 = trained (assistant response), -1 = padding
    mask =      [0, 0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,
                 0, 0,
                 1,1,1,1,1,1,1,1,1,1,1,
                 1]

    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(n * 0.7, 16), 3.2))

    for i, (tok, m) in enumerate(zip(tokens, mask)):
        color = TRAIN_COLOR if m == 1 else MASK_COLOR
        rect = mpatches.FancyBboxPatch(
            (i, 0), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='#999999', linewidth=0.8
        )
        ax.add_patch(rect)
        # Token label
        label = tok if len(tok) <= 4 else tok
        fontsize = 7.5 if len(tok) > 3 else 9
        ax.text(i + 0.45, 0.45, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if m == 1 else 'normal',
                color='white' if m == 1 else '#666666')

    # Brace labels
    user_end = 18  # index of <|asst|> (last masked)
    asst_start = 19
    asst_end = n - 1

    ax.annotate('', xy=(0, -0.25), xytext=(user_end + 0.9, -0.25),
                arrowprops=dict(arrowstyle='-', color='#999999', lw=1.5))
    ax.text((user_end + 0.9) / 2, -0.55, 'MASKED — no loss, no gradient',
            ha='center', va='center', fontsize=10, color='#888888')

    ax.annotate('', xy=(asst_start, -0.25), xytext=(asst_end + 0.9, -0.25),
                arrowprops=dict(arrowstyle='-', color=TRAIN_COLOR, lw=1.5))
    ax.text((asst_start + asst_end + 0.9) / 2, -0.55, 'TRAINED — loss computed, gradient flows',
            ha='center', va='center', fontsize=10, color=TRAIN_COLOR, fontweight='bold')

    # Legend
    masked_patch = mpatches.Patch(color=MASK_COLOR, label='Masked (label = -100)')
    trained_patch = mpatches.Patch(color=TRAIN_COLOR, label='Trained (actual label)')
    ax.legend(handles=[masked_patch, trained_patch], loc='upper right',
              fontsize=9, framealpha=0.9)

    ax.set_xlim(-0.3, n + 0.3)
    ax.set_ylim(-1.0, 1.5)
    ax.axis('off')
    ax.set_title('Loss Mask: Only the Assistant\'s Response Tokens Are Trained', pad=15)

    plt.savefig(os.path.join(OUT_DIR, 'sft_loss_mask.png'))
    plt.close()
    print("✓ sft_loss_mask.png")


# ─────────────────────────────────────────────────────────
# 2. SFT Training Loss Curve
# ─────────────────────────────────────────────────────────
def generate_training_loss_curve():
    """Show SFT loss dropping over training steps."""
    np.random.seed(42)
    steps = np.arange(1000)
    # Simulate realistic SFT loss curve: fast initial drop, then gradual
    base_loss = 4.0 * np.exp(-steps / 150) + 0.3
    noise = np.random.normal(0, 0.15, len(steps)) * np.exp(-steps / 300)
    raw_loss = base_loss + noise
    raw_loss = np.clip(raw_loss, 0.1, 5.0)

    # Smoothed version
    window = 50
    smoothed = np.convolve(raw_loss, np.ones(window)/window, mode='valid')
    smooth_steps = steps[:len(smoothed)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, raw_loss, alpha=0.2, color=TRAIN_COLOR, linewidth=0.8, label='Raw loss')
    ax.plot(smooth_steps, smoothed, color=TRAIN_COLOR, linewidth=2.5, label='Smoothed (window=50)')

    # Annotations
    ax.annotate('Model learns chat format quickly\n(~200 steps)',
                xy=(200, smoothed[200]), xytext=(350, 2.5),
                arrowprops=dict(arrowstyle='->', color='#666666'),
                fontsize=10, color='#666666')
    ax.annotate(f'Final loss ≈ {smoothed[-1]:.1f}\nMemorizing specific responses',
                xy=(900, smoothed[-1]), xytext=(650, 1.2),
                arrowprops=dict(arrowstyle='->', color='#666666'),
                fontsize=10, color='#666666')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss (assistant tokens only)')
    ax.set_title('SFT Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.5)

    plt.savefig(os.path.join(OUT_DIR, 'sft_training_loss.png'))
    plt.close()
    print("✓ sft_training_loss.png")


# ─────────────────────────────────────────────────────────
# 3. Catastrophic Forgetting Bar Chart
# ─────────────────────────────────────────────────────────
def generate_forgetting_chart():
    """Bar chart: base model vs SFT model on Shakespeare validation loss."""
    fig, ax = plt.subplots(figsize=(7, 5))

    models = ['Base Model\n(pre-trained)', 'SFT Model\n(after fine-tuning)']
    losses = [2.12, 2.31]
    colors = ['#6CB86C', ACCENT]

    bars = ax.bar(models, losses, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{loss:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Difference annotation
    ax.annotate('', xy=(1, 2.12), xytext=(1, 2.31),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax.text(1.35, 2.215, '+0.19\n(worse at\nShakespeare)',
            fontsize=10, color=ACCENT, va='center')

    ax.set_ylabel('Shakespeare Validation Loss')
    ax.set_title('Catastrophic Forgetting: The Cost of New Behavior')
    ax.set_ylim(0, 2.8)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'sft_catastrophic_forgetting.png'))
    plt.close()
    print("✓ sft_catastrophic_forgetting.png")


# ─────────────────────────────────────────────────────────
# 4. Learning Rate Comparison
# ─────────────────────────────────────────────────────────
def generate_lr_comparison():
    """Show what happens with different learning rates during SFT."""
    np.random.seed(42)
    steps = np.arange(1000)

    # Good LR (1e-4): smooth convergence
    good_base = 4.0 * np.exp(-steps / 150) + 0.3
    good_noise = np.random.normal(0, 0.08, len(steps))
    good_loss = good_base + good_noise

    # Too high (3e-4): fast drop but unstable, forgetting spikes
    np.random.seed(43)
    high_base = 4.0 * np.exp(-steps / 80) + 0.15
    high_noise = np.random.normal(0, 0.25, len(steps))
    # Add instability spikes
    for i in range(5):
        spike_pos = np.random.randint(200, 800)
        high_base[spike_pos:spike_pos+30] += np.random.uniform(0.5, 1.5)
    high_loss = high_base + high_noise
    high_loss = np.clip(high_loss, 0.05, 5.0)

    # Too low (1e-5): barely learns
    np.random.seed(44)
    low_base = 4.0 * np.exp(-steps / 800) + 1.5
    low_noise = np.random.normal(0, 0.05, len(steps))
    low_loss = low_base + low_noise

    # Smooth all
    window = 30
    def smooth(arr):
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 5))

    s = steps[:len(smooth(good_loss))]
    ax.plot(s, smooth(high_loss), color=ACCENT, linewidth=2, label='lr=3e-4 (too high) — unstable, destroys knowledge')
    ax.plot(s, smooth(good_loss), color=TRAIN_COLOR, linewidth=2.5, label='lr=1e-4 (good) — smooth convergence')
    ax.plot(s, smooth(low_loss), color='#FFA500', linewidth=2, label='lr=1e-5 (too low) — barely learns the format')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('SFT Loss')
    ax.set_title('Learning Rate Selection: The Fine-Tuning Tradeoff')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'sft_learning_rate_comparison.png'))
    plt.close()
    print("✓ sft_learning_rate_comparison.png")


# ─────────────────────────────────────────────────────────
# 5. With vs Without Loss Masking
# ─────────────────────────────────────────────────────────
def generate_masking_comparison():
    """Show the effect of masking: what the model learns to generate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Without masking
    ax = axes[0]
    ax.set_title('WITHOUT Loss Masking', fontsize=13, fontweight='bold', color=ACCENT)

    rows = [
        ('User:', '"Write a greeting"', MASK_COLOR),
        ('Model:', '"User: Tell me about love\\nAssistant: Love is..."', '#FFD4D4'),
        ('Problem:', 'Model learns to GENERATE user messages too', '#FFD4D4'),
        ('Result:', '50% of gradient wasted on wrong task', '#FFD4D4'),
    ]
    for i, (label, text, color) in enumerate(rows):
        y = 0.85 - i * 0.22
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top')
        ax.text(0.20, y, text, transform=ax.transAxes, fontsize=10,
                va='top', style='italic' if i > 0 else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))
    ax.axis('off')

    # Right: With masking
    ax = axes[1]
    ax.set_title('WITH Loss Masking', fontsize=13, fontweight='bold', color='#2E7D32')

    rows = [
        ('User:', '"Write a greeting"', MASK_COLOR),
        ('Model:', '"Good morrow to thee, noble friend!"', '#D4EED4'),
        ('Why:', 'Model only learns to RESPOND', '#D4EED4'),
        ('Result:', '100% of gradient on the task that matters', '#D4EED4'),
    ]
    for i, (label, text, color) in enumerate(rows):
        y = 0.85 - i * 0.22
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top')
        ax.text(0.18, y, text, transform=ax.transAxes, fontsize=10,
                va='top', style='italic' if i > 0 else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))
    ax.axis('off')

    plt.suptitle('Loss Masking: Focus the Gradient on What Matters',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sft_masking_comparison.png'))
    plt.close()
    print("✓ sft_masking_comparison.png")


# ─────────────────────────────────────────────────────────
# 6. Data Quantity vs Quality
# ─────────────────────────────────────────────────────────
def generate_data_scaling():
    """Show diminishing returns of more data and the effect of quality filtering."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # X axis: dataset size (log scale)
    sizes = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])

    # All data (including easy examples): improves then plateaus/degrades
    all_data_quality = np.array([30, 55, 68, 78, 80, 79, 77])

    # Filtered/curated data: keeps improving
    filtered_quality = np.array([30, 60, 74, 85, 88, 90, 91])

    ax.semilogx(sizes, all_data_quality, 'o-', color=ACCENT, linewidth=2,
                markersize=8, label='All synthetic data (unfiltered)')
    ax.semilogx(sizes, filtered_quality, 's-', color=TRAIN_COLOR, linewidth=2,
                markersize=8, label='Quality-filtered data')

    # Annotations
    ax.annotate('Diminishing returns —\neasy examples dilute signal',
                xy=(50000, 79), xytext=(15000, 65),
                arrowprops=dict(arrowstyle='->', color='#666666'),
                fontsize=10, color=ACCENT)
    ax.annotate('Filtering keeps only\nhard, useful examples',
                xy=(50000, 90), xytext=(12000, 95),
                arrowprops=dict(arrowstyle='->', color='#666666'),
                fontsize=10, color=TRAIN_COLOR)

    ax.set_xlabel('Number of SFT Examples')
    ax.set_ylabel('Downstream Task Quality (%)')
    ax.set_title('Data Scaling: More Isn\'t Always Better')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(20, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'sft_data_scaling.png'))
    plt.close()
    print("✓ sft_data_scaling.png")


if __name__ == '__main__':
    generate_loss_mask_heatmap()
    generate_training_loss_curve()
    generate_forgetting_chart()
    generate_lr_comparison()
    generate_masking_comparison()
    generate_data_scaling()
    print("\nAll SFT diagrams generated!")
