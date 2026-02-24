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


# ─────────────────────────────────────────────────────────
# 7. Single Forward Pass
# ─────────────────────────────────────────────────────────
def generate_single_forward_pass():
    """Show that one token in → N layers → one probability distribution out."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw the token sequence on the left
    tokens = ['The', 'cat', 'sat', 'on', '???']
    for i, tok in enumerate(tokens):
        color = TRAIN_COLOR if i < 4 else ACCENT
        alpha = 0.4 if i < 3 else 1.0
        rect = mpatches.FancyBboxPatch(
            (0.5 + i * 1.4, 3.5), 1.1, 0.8,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor='#666', linewidth=1, alpha=alpha
        )
        ax.add_patch(rect)
        ax.text(1.05 + i * 1.4, 3.9, tok, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', alpha=alpha if i < 3 else 1)

    # Arrow from "on" into the stack
    ax.annotate('', xy=(5.5, 2.9), xytext=(5.5, 3.45),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(5.5, 3.15, '"on"', ha='center', va='center', fontsize=9, color='#666')

    # Draw N-layer transformer stack
    n_layers = 6
    stack_x = 3.8
    stack_w = 3.4
    for i in range(n_layers):
        y = 2.6 - i * 0.4
        shade = 0.3 + i * 0.1
        rect = mpatches.FancyBboxPatch(
            (stack_x, y), stack_w, 0.32,
            boxstyle="round,pad=0.05",
            facecolor=plt.cm.Blues(shade), edgecolor='#999', linewidth=0.8
        )
        ax.add_patch(rect)
        label = f'Layer {i+1}' if i < 2 or i == n_layers - 1 else ('...' if i == 3 else '')
        if label:
            ax.text(stack_x + stack_w / 2, y + 0.16, label,
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Arrow out of the stack
    ax.annotate('', xy=(5.5, -0.15), xytext=(5.5, 0.25),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Output probability distribution (simplified bar chart)
    out_words = ['the', 'a', 'his', 'my', 'her']
    out_probs = [0.45, 0.2, 0.15, 0.1, 0.1]
    bar_x_start = 3.8
    bar_width = 0.55
    for i, (w, p) in enumerate(zip(out_words, out_probs)):
        x = bar_x_start + i * (bar_width + 0.12)
        bar_h = p * 1.8
        color = ACCENT if i == 0 else '#BBBBBB'
        rect = mpatches.FancyBboxPatch(
            (x, -0.2 - bar_h), bar_width, bar_h,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor='#999', linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(x + bar_width / 2, -0.25 - bar_h, f'{w}\n{p:.0%}',
                ha='center', va='top', fontsize=7.5, color='#444')

    # Labels
    ax.text(5.5, 4.6, 'One token = one forward pass through the entire stack',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#333')
    ax.text(5.5, -1.6, 'Output: probability distribution over vocabulary',
            ha='center', va='center', fontsize=10, color='#666')
    ax.text(1.5, 4.55, 'Context so far', ha='center', va='center',
            fontsize=9, color='#888')

    # Side note
    ax.text(9.2, 1.5, 'Same N layers,\nsame compute,\nevery single token.\n\nNo way to\n"think harder."',
            ha='center', va='center', fontsize=10, color=ACCENT,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3F3', edgecolor=ACCENT, alpha=0.8))

    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-2.2, 5.0)
    ax.axis('off')

    plt.savefig(os.path.join(OUT_DIR, 'sft_single_forward_pass.png'))
    plt.close()
    print("✓ sft_single_forward_pass.png")


# ─────────────────────────────────────────────────────────
# 8. Tokens as Computation (answer-first vs think-first)
# ─────────────────────────────────────────────────────────
def generate_tokens_as_computation():
    """Side-by-side: answer-first (wrong) vs think-first (right) with apple/orange example."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def draw_approach(ax, title, title_color, tokens, colors, result, result_color, annotation):
        ax.set_title(title, fontsize=13, fontweight='bold', color=title_color, pad=15)

        y = 0.92
        for i, (tok, col) in enumerate(zip(tokens, colors)):
            ax.text(0.05, y - i * 0.085, tok, transform=ax.transAxes, fontsize=10,
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=col, alpha=0.7, edgecolor='#ccc'))

        # Result box
        ax.text(0.5, 0.05, result, transform=ax.transAxes, fontsize=12,
                ha='center', va='center', fontweight='bold', color=result_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=result_color, alpha=0.15, edgecolor=result_color))

        # Side annotation
        ax.text(0.95, 0.5, annotation, transform=ax.transAxes, fontsize=9,
                ha='right', va='center', color='#888', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f9f9f9', edgecolor='#ddd'))

        ax.axis('off')

    # Left: Answer-first (WRONG)
    left_tokens = [
        'Q: I have 3 apples and 2 oranges. How many fruits?',
        'A: 4',  # wrong — committed answer before thinking
        '',
        '(The model already output "4" — no room to compute)',
    ]
    left_colors = ['#E8E8E8', '#FFD4D4', '#FFFFFF', '#FFF8F0']

    draw_approach(ax1, 'Answer-First: WRONG', ACCENT,
                  left_tokens, left_colors,
                  'Answer: 4  ✗', ACCENT,
                  '1 forward pass\nfor the answer.\n\nNo computation\nhappened yet!')

    # Right: Think-first (CORRECT)
    right_tokens = [
        'Q: I have 3 apples and 2 oranges. How many fruits?',
        'A: Let me think step by step.',
        '• Apples: 3',
        '• Oranges: 2',
        '• Total: 3 + 2 = 5',
        'The answer is 5.',
    ]
    right_colors = ['#E8E8E8', '#D4E8D4', '#D4E8D4', '#D4E8D4', '#D4E8D4', '#C4DFC4']

    draw_approach(ax2, 'Think-First: CORRECT', '#2E7D32',
                  right_tokens, right_colors,
                  'Answer: 5  ✓', '#2E7D32',
                  '5 forward passes\nfor intermediate\nsteps.\n\nEach token =\nmore compute!')

    plt.suptitle('Tokens as Computation: The Model Needs Tokens to Think',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sft_tokens_as_computation.png'))
    plt.close()
    print("✓ sft_tokens_as_computation.png")


# ─────────────────────────────────────────────────────────
# 9. Jagged Intelligence Profile
# ─────────────────────────────────────────────────────────
def generate_jagged_intelligence():
    """Horizontal bar chart: superhuman at some tasks, failing at others."""
    fig, ax = plt.subplots(figsize=(12, 7))

    tasks = [
        'Text summarization',
        'Translation',
        'Code generation',
        'Legal analysis',
        'Math word problems',
        'Trivia / factual recall',
        'Spatial reasoning',
        'Counting characters',
        'Multi-digit multiplication',
        'Spelling / anagrams',
    ]
    # Relative capability score (0-100, where 50 = average human)
    scores = [92, 90, 85, 80, 72, 65, 30, 15, 20, 10]
    tasks = tasks[::-1]
    scores = scores[::-1]

    colors = []
    for s in scores:
        if s >= 75:
            colors.append('#2E7D32')  # green — superhuman
        elif s >= 50:
            colors.append(TRAIN_COLOR)  # blue — above average
        elif s >= 35:
            colors.append('#FFA500')  # orange — below average
        else:
            colors.append(ACCENT)  # red — failing

    bars = ax.barh(tasks, scores, color=colors, height=0.6, edgecolor='white', linewidth=1)

    # Human baseline line
    ax.axvline(x=50, color='#333', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(51, len(tasks) - 0.3, 'Average human', fontsize=9, color='#555',
            va='bottom', style='italic')

    # Score labels
    for bar, score in zip(bars, scores):
        x = bar.get_width() + 1.5
        ax.text(x, bar.get_y() + bar.get_height() / 2, f'{score}',
                va='center', fontsize=10, fontweight='bold', color='#444')

    # Legend
    legend_items = [
        mpatches.Patch(color='#2E7D32', label='Superhuman (75+)'),
        mpatches.Patch(color=TRAIN_COLOR, label='Above average (50-74)'),
        mpatches.Patch(color='#FFA500', label='Below average (35-49)'),
        mpatches.Patch(color=ACCENT, label='Failing (< 35)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9, framealpha=0.9)

    ax.set_xlabel('Capability Score (50 = average human)', fontsize=11)
    ax.set_title('Jagged Intelligence: The Same Model, Wildly Different Abilities',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sft_jagged_intelligence.png'))
    plt.close()
    print("✓ sft_jagged_intelligence.png")


if __name__ == '__main__':
    generate_loss_mask_heatmap()
    generate_training_loss_curve()
    generate_forgetting_chart()
    generate_lr_comparison()
    generate_masking_comparison()
    generate_data_scaling()
    generate_single_forward_pass()
    generate_tokens_as_computation()
    generate_jagged_intelligence()
    print("\nAll SFT diagrams generated!")
