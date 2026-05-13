import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # NASEM Future of Drought Figure 4-1

    Conceptual figure showing how USDM drought classes map onto a
    precipitation CDF, and how a shift in baseline climate changes
    the precipitation thresholds for each class.

    Two synthetic Weibull CDFs represent a wetter (1st) and drier
    (2nd) baseline. No real data — purely illustrative.

    *Justin Mankin -- 2026*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    return Line2D, mo, np, plt


@app.cell
def _(np):
    """Synthetic Weibull CDFs for the two baseline climates."""
    x = np.linspace(0, 160, 2000)
    k1, lam1 = 2.6, 156    # 1st baseline (wetter)
    k2, lam2 = 2.6, 120    # 2nd baseline (drier, shifted left)
    cdf1 = 1 - np.exp(-(x / lam1) ** k1)
    cdf2 = 1 - np.exp(-(x / lam2) ** k2)
    return cdf1, cdf2, x


@app.cell
def _(cdf1, cdf2, np, x):
    """Compute precipitation thresholds for each USDM class on both baselines."""
    usdm_classes = [
        ('D4', 0.02, '#730000'),
        ('D3', 0.05, '#CC0000'),
        ('D2', 0.10, '#E05500'),
        ('D1', 0.20, '#FFA500'),
        ('D0', 0.30, '#DDCC00'),
    ]

    def precip_at_prob(cdf_vals, x_vals, prob):
        idx = np.searchsorted(cdf_vals, prob)
        if idx == 0:
            return x_vals[0]
        if idx >= len(x_vals):
            return x_vals[-1]
        f = (prob - cdf_vals[idx - 1]) / (cdf_vals[idx] - cdf_vals[idx - 1])
        return x_vals[idx - 1] + f * (x_vals[idx] - x_vals[idx - 1])

    thresholds = []
    for _name, _prob, _color in usdm_classes:
        _p1 = precip_at_prob(cdf1, x, _prob)
        _p2 = precip_at_prob(cdf2, x, _prob)
        thresholds.append((_name, _prob, _color, _p1, _p2))
    return (thresholds,)


@app.cell
def _(Line2D, cdf1, cdf2, np, plt, thresholds, x):
    """Build the schematic figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#E0E0E0')

    # Plot both CDFs
    ax.plot(x, cdf1, color='black', lw=4.5, solid_capstyle='round', zorder=10)
    ax.plot(x, cdf2, color='black', lw=4.5, dashes=[14, 7],
            dash_capstyle='round', zorder=10)

    # Draw each USDM class: threshold lines, labels, and shift arrows
    for cls_name, prob, color, p1, p2 in thresholds:
        lw_line = 2.5 if cls_name == 'D0' else 2.0

        ax.plot([0, p1], [prob, prob], color=color, lw=lw_line,
                solid_capstyle='round', zorder=12)
        ax.plot([p1, p1], [0, prob], color=color, lw=lw_line,
                solid_capstyle='round', zorder=12)

        mid_x = (p1 + p2) / 2
        ax.text(mid_x, prob + 0.012, cls_name, fontsize=13,
                fontweight='bold', color=color, ha='center', va='bottom',
                zorder=15,
                bbox=dict(facecolor='#E0E0E0', edgecolor='none', pad=1.5))

        # Leftward arrow: same percentile, precip threshold decreases
        ax.annotate('', xy=(p2, prob - 0.004), xytext=(p1, prob - 0.004),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.0, mutation_scale=8), zorder=13)

        # Upward arrow: same precip, probability increases on 2nd baseline
        prob2_at_p1 = float(np.interp(p1, x, cdf2))
        ax.annotate('', xy=(p1 + 0.8, prob2_at_p1), xytext=(p1 + 0.8, prob),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.0, mutation_scale=8), zorder=13)

    # Y-axis: colored probability labels
    yvals = [0] + [prob for _, prob, _, _, _ in thresholds]
    ax.set_yticks(sorted(yvals))
    ylabels = []
    for yv in sorted(yvals):
        if yv == 0:
            ylabels.append('0')
        elif yv >= 0.1:
            ylabels.append('%.1f' % yv)
        else:
            ylabels.append('%.2f' % yv)
    ax.set_yticklabels(ylabels, fontweight='bold', fontsize=10)
    color_map = {prob: color for _, prob, color, _, _ in thresholds}
    for label in ax.get_yticklabels():
        val = float(label.get_text())
        if val in color_map:
            label.set_color(color_map[val])

    # X-axis: colored threshold values
    xtick_vals = sorted([p1 for _, _, _, p1, _ in thresholds])
    xtick_colors = {}
    for _, _, color, p1, _ in thresholds:
        xtick_colors[p1] = color
    ax.set_xticks([0] + xtick_vals)
    xlabels_x = ['0'] + ['%d' % round(v) for v in xtick_vals]
    ax.set_xticklabels(xlabels_x, fontweight='bold', fontsize=10)
    for label in ax.get_xticklabels():
        txt = label.get_text()
        if txt == '0':
            continue
        val = int(txt)
        closest = min(xtick_vals, key=lambda v: abs(round(v) - val))
        label.set_color(xtick_colors[closest])

    ax.set_xlim(0, 130)
    ax.set_ylim(0, 0.34)
    ax.set_xlabel('WY PRECIPITATION [MM]', fontsize=12, fontweight='bold')
    ax.set_ylabel('CUMULATIVE PROBABILITY [FRAC.]', fontsize=12, fontweight='bold')
    ax.set_title('DROUGHT CLASSIFICATION & NONSTATIONARITY',
                 fontsize=13, fontweight='bold', pad=12)

    legend_handles = [
        Line2D([0], [0], color='black', lw=4, solid_capstyle='round',
               label='1ST BASELINE'),
        Line2D([0], [0], color='black', lw=4, dashes=[3, 1.5],
               dash_capstyle='round', label='2ND BASELINE'),
    ]
    legend = ax.legend(handles=legend_handles, loc='lower right', fontsize=11,
                       framealpha=0.9, edgecolor='#666666', fancybox=False,
                       handlelength=3)
    legend.set_zorder(20)  
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
