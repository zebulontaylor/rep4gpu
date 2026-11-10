"""Plot retained inference measurements; run from the repository root."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('Decode · floor', 'final_device_floor_decode.json'),
    ('Decode · probabilistic', 'final_device_prob_decode.json'),
    ('Prefill · floor', 'final_device_floor_prefill.json'),
    ('Prefill · probabilistic', 'final_device_prob_prefill.json'),
]


def main():
    rows = [json.loads((ROOT/'results'/name).read_text())['results'][0]
            for _, name in CASES]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    positions = np.arange(len(rows))
    for offset, key, label, color in [(-.19, 'fp32_graph_ms', 'FP32 PyTorch', '#147d92'),
                                     (.19, 'fp16_graph_ms', 'FP16 PyTorch', '#b85d30')]:
        values = [r['ideal_slowdowns'][key] for r in rows]
        bars = ax.bar(positions+offset, values, width=.36, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}×' for v in values], padding=4, fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(5, 250)
    for threshold in (10, 100):
        ax.axhline(threshold, color='#555555', linestyle='--', linewidth=1, zorder=0)
        ax.text(3.65, threshold, f'{threshold}×', va='bottom', ha='right', color='#555555')
    ax.set_xticks(positions, [label for label, _ in CASES])
    ax.set_ylabel('Slowdown versus resident CUDA-graph inference')
    ax.set_title('Secret 2048 → 8192 ReLU → 2048 inference', loc='left', weight='bold', pad=20)
    ax.legend(loc='upper left', frameon=False)
    fig.supxlabel('Zero-network estimate; RTX 4060 Laptop · decode batch 1 / prefill batch 128\n'
                  'Private weights already shared · Q14 bounded rounding · CUDA ChaCha20/BLAKE3 variant',
                  fontsize=9)
    output = ROOT/'docs'/'assets'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/'performance.png', dpi=180)
    fig.savefig(output/'performance.svg')


if __name__ == '__main__':
    main()
