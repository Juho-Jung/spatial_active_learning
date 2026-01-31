import matplotlib.pyplot as plt
import numpy as np

# ---------- Data ----------
budgets = [10, 20, 50, 70, 100]
methods = [
    "Random", "Area-Random", "Uncertainty", "Area-Uncertainty", "Diversity",
    "Coreset", "TAUDIS", "USIM", "Loss-prediction", "SPARCL (ours)"
]

dice = np.array([
    [0.0309, 0.1503, 0.2507, 0.3454, 0.4056],
    [0.0078, 0.2043, 0.2940, 0.3267, 0.4056],
    [0.0951, 0.2305, 0.3249, 0.3752, 0.4056],
    [0.0550, 0.2732, 0.3403, 0.3597, 0.4056],
    [0.0173, 0.2252, 0.2966, 0.3437, 0.4056],
    [0.0106, 0.2542, 0.3479, 0.3417, 0.4056],
    [0.0351, 0.2490, 0.3358, 0.3528, 0.3968],
    [0.1385, 0.2875, 0.3391, 0.3571, 0.3945],
    [0.0132, 0.2264, 0.3270, 0.3805, 0.4060],
    [0.2195, 0.2986, 0.3720, 0.3821, 0.3960]
])

csp = np.array([
    [1.5078, 1.5153, 1.4797, 1.3408, 1.3095],
    [1.4111, 1.5413, 1.4807, 1.4280, 1.3095],
    [1.7369, 1.4394, 1.3077, 1.3015, 1.3095],
    [2.49, 1.5131, 1.3801, 1.3087, 1.3095],
    [3.352, 1.6797, 1.3991, 1.3016, 1.3095],
    [1.3886, 1.5239, 1.2481, 1.3457, 1.3095],
    [1.3748, 1.4350, 1.3564, 1.3947, 1.2375],
    [1.4117, 1.4216, 1.2888, 1.3572, 1.2192],
    [1.4088, 1.6443, 1.4255, 1.2358, 1.3309],
    [1.3939, 1.3743, 1.2602, 1.2067, 1.1088]
])

cperf = np.array([
    [0.1089, 0.2803, 0.2841, 0.4160, 0.4762],
    [0.0288, 0.2395, 0.3149, 0.3379, 0.4762],
    [0.2073, 0.3169, 0.3879, 0.4064, 0.4762],
    [0.0774, 0.3036, 0.3658, 0.4100, 0.4762],
    [0.0514, 0.2220, 0.3487, 0.4191, 0.4762],
    [0.0341, 0.2897, 0.3925, 0.3846, 0.4762],
    [0.2932, 0.2937, 0.3682, 0.3507, 0.4601],
    [0.3374, 0.3056, 0.3874, 0.3640, 0.4603],
    [0.0524, 0.2476, 0.3696, 0.4296, 0.4603],
    [0.3059, 0.3582, 0.4231, 0.4202, 0.5051]
])

# ---------- Style ----------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 22,   # BIGGER TITLES for Overleaf
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

tab10 = list(plt.get_cmap("tab10").colors)
color_map = {
    "Random": tab10[0],
    "Area-Random": tab10[1],
    "Uncertainty": tab10[2],
    "Area-Uncertainty": tab10[3],
    "Diversity": tab10[4],
    "Coreset": tab10[5],
    "TAUDIS": tab10[6],
    "USIM": tab10[7],
    "Loss-prediction": tab10[8],
    "SPARCL (ours)": tab10[9],
}

markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>']
titles = [r"Dice $\uparrow$", r"$C_{\mathrm{spatial}} \uparrow$", r"$C_{\mathrm{perf}} \uparrow$"]
datasets = [dice, csp, cperf]

fig, axs = plt.subplots(1, 3, figsize=(18, 4))

handles = []
labels = []

for j, (ax, data, title) in enumerate(zip(axs, datasets, titles)):
    for i, name in enumerate(methods):
        lw = 3 if name == "SPARCL (ours)" else 1.8
        ls = "--" if name == "SPARCL (ours)" else "-"
        line = ax.plot(
            budgets, data[i],
            marker=markers[i], markersize=8,
            color=color_map[name],
            linewidth=lw, linestyle=ls,
            label=name
        )[0]
        if j == 0:
            handles.append(line)
            labels.append(name)

    # Title bigger + bold
    ax.set_title(title, fontweight="bold")

    # Left graph only keeps Y label
    if j == 0:
        ax.set_ylabel("Value", fontsize=18, fontweight="bold")
    else:
        ax.set_ylabel("")  # REMOVE middle/right y-axis label

    ax.set_xlabel("Budget (%)", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.subplots_adjust(wspace=0.18)

# Bigger legend
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.22),
    frameon=True,
    prop={'size': 14}
)

# Save PDF + PNG
fig.savefig("alc_miccai_overleaf_final.pdf", format="pdf", dpi=300, bbox_inches="tight")
fig.savefig("alc_miccai_overleaf_final.png", dpi=300, bbox_inches="tight")

print("Saved PDF and PNG!")
