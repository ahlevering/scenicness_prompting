import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the file paths
results_path_25 = "data/outputs/contrastive/clip/zero_shot_contrasts_25.csv"
results_path_full = "data/outputs/contrastive/clip/zero_shot_contrasts.csv"

# Read the CSV files into pandas DataFrames
df_25 = pd.read_csv(results_path_25)
df_full = pd.read_csv(results_path_full)

# Define the data column and color palette
data_column = "kendalls_tau"
palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

# Set the style for the violin plots
sns.set_style("whitegrid")

# Plot the violin plots for the full dataset
sns.violinplot(data=df_full, x="context", y=data_column, palette=palette, inner=None, ax=axes[0], cut=0)
axes[0].set_ylabel("Kendall's $\\tau$", fontsize=14)
axes[0].set_title("Prompt context performance: full dataset", fontweight="bold", fontsize=16, pad=10)
axes[0].tick_params(axis="both", which="major", labelsize=12)
axes[0].spines["right"].set_visible(False)
axes[0].spines["top"].set_visible(False)
axes[0].set_ylim(-0.2, 0.55)
axes[0].set_xticklabels([])
axes[0].set_xlabel("")
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Plot the violin plots for 25 samples
sns.violinplot(data=df_25, x="context", y=data_column, palette=palette, inner=None, ax=axes[1], cut=0)
axes[1].set_ylabel("Kendall's $\\tau$", fontsize=14)
axes[1].set_title("Prompt context performance: 25 samples", fontweight="bold", fontsize=16, pad=10)
axes[1].tick_params(axis="both", which="major", labelsize=12)
axes[1].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].set_ylim(-0.35, 0.75)
axes[1].set_xticklabels([])
axes[1].set_xlabel("")
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Add lines at y-axis ticks
for a in [0, 1]:
    for y in axes[a].get_yticks():
        axes[a].axhline(y=y, color="gray", linestyle="--", linewidth=0.5, zorder=0)


# Create a shared legend for both subplots
legend_handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=10) for color in palette
]
legend_labels = df_25["context"].unique()
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.15),
    title="Prompt Context",
    prop={"size": 12},
)
legend.set_title("", prop={"size": 0, "weight": "bold"})  # Empty title

# Adjust the layout to reduce whitespace
fig.tight_layout()

# Save the plot as an image file
filename = "zeroshot_comparison.png"
plt.savefig(filename, bbox_inches="tight", dpi=300)
plt.close()

### FEW SHOT METRICS ###
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot_line(data, metric, ax):
    sns.lineplot(
        data=data[data["Model"] != "Baseline"],
        x="Samples",
        y=metric,
        hue="Model",
        style="Model",
        palette=["blue", "red"],
        linewidth=1.5,
        ax=ax,
        markers=["o", "o"],
        ci=None,
    )
    baseline_metric = data.loc[data["Model"] == "Baseline", metric].values[0]  # Get the baseline metric value
    ax.axhline(
        y=baseline_metric, linestyle="-", color="black", linewidth=2, label="Baseline"
    )  # Add the solid black line for baseline
    ax.set(xlabel="Samples", ylabel=metric)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))


# Define the table data
# SigLIP
# data = {
#     "Model": [
#         "ConvNeXt-L",
#         "ConvNeXt-L",
#         "ConvNeXt-L",
#         "ConvNeXt-L",
#         "ConvNeXt-L",
#         "ViT-14/L",
#         "ViT-14/L",
#         "ViT-14/L",
#         "ViT-14/L",
#         "ViT-14/L",
#         "Baseline",
#     ],
#     "Samples": [25, 50, 100, 250, 500, 25, 50, 100, 250, 500, 25],
#     "RMSE": [1.228, 1.249, 1.159, 1.064, 1.042, 1.232, 1.147, 1.049, 0.948, 0.905, 0.8037],
#     "R2": [0.641, 0.632, 0.694, 0.747, 0.761, 0.66, 0.712, 0.756, 0.805, 0.824, 0.8642],
#     "Tau": [0.465, 0.453, 0.508, 0.550, 0.561, 0.48, 0.516, 0.561, 0.607, 0.624, 0.6679]
# }

# CLIP
data = {
    "Model": [
        "ConvNeXt-L",
        "ConvNeXt-L",
        "ConvNeXt-L",
        "ConvNeXt-L",
        "ConvNeXt-L",
        "ViT-14/L",
        "ViT-14/L",
        "ViT-14/L",
        "ViT-14/L",
        "ViT-14/L",
        "Baseline",
    ],
    "Samples": [25, 50, 100, 250, 500, 25, 50, 100, 250, 500, 25],
    "RMSE": [1.228, 1.249, 1.159, 1.064, 1.042, 1.232, 1.125, 1.033, 0.948, 0.905, 0.8037],
    "R2": [0.641, 0.632, 0.694, 0.747, 0.761, 0.66, 0.718, 0.767, 0.805, 0.824, 0.8642],
    "Tau": [0.465, 0.453, 0.508, 0.550, 0.561, 0.48, 0.52, 0.571, 0.607, 0.624, 0.6679]
}


# Convert the table data to a pandas DataFrame
df = pd.DataFrame(data)

# Create the figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))  # Adjusted figsize for a shorter y-axis

# Plot line graphs for each metric
plot_line(df, "RMSE", axes[0])
plot_line(df, "R2", axes[1])
plot_line(df, "Tau", axes[2])

# Set individual titles for each subplot
axes[0].set_title("RMSE", fontweight="bold")
axes[1].set_title(r"$R^2$", fontweight="bold")
axes[2].set_title("Kendall's $\\tau$", fontweight="bold")

# Set x-axis ticks based on the number of samples
samples = sorted(df["Samples"].unique())
for ax in axes:
    ax.set_xticks(samples)
    ax.set_xticklabels([int(sample) for sample in samples], rotation=0, ha="center", fontsize="small")

# Remove legends from subplots
for ax in axes:
    ax.legend().remove()

handles, labels = axes[0].get_legend_handles_labels()
handles = handles[:-1]  # Exclude the baseline handle
labels = labels[:-1]  # Exclude the baseline label
handles.insert(0, Line2D([0], [0], color="black", linewidth=2))
labels.insert(0, "ViT-14/L (full dataset)")
handles.insert(1, Line2D([0], [0], color="magenta", marker="*", linestyle="None"))
labels.insert(1, "Contrastive prompting (finetuned)")
fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.115))

# Change y-axis label for R2 to include superscript
axes[1].set_ylabel(r"$R^2$", fontsize=12)

# Add magenta point to all subplots
magenta_point = {"Samples": [25], "RMSE": [1.956], "R2": [0.6783], "Tau": [0.4973]}
for ax, metric in zip(axes, df.columns[2:]):
    ax.scatter(magenta_point["Samples"], magenta_point[metric], marker="*", color="magenta")

    # Adjust the y-axis limits to include the magenta point
    ymin, ymax = ax.get_ylim()
    y_val = magenta_point[metric][0]
    ax.set_ylim(min(ymin, y_val - 0.1), max(ymax, y_val + 0.1))

# Add gray gridlines to the subplots
for ax in axes:
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

# Adjust spacing between the suptitle and subplots
fig.tight_layout(pad=2)

# Save the plot as an image file
filename = "metric_comparisons.png"
plt.savefig(filename)
plt.close()

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define the folder path and other parameters
folder_path = "data/maps"
output_path = "data/outputs/collage.png"
num_rows = 2
num_cols = 3
legend_height = 50

# Load and crop the maps
maps = []
map_titles = {
    "gt.png": "ScenicOrNot reference",
    "zero_shot_contrastive_25.png": "Zero-shot contrastive\n(best prompt from 25 samples)",
    "few_shot_25.png": "ViT-14/L Linear probe\n(25 samples)",
    "few_shot_500.png": "ViT-14/L Linear probe\n(500 samples)",
    "early_ensembling.png": "Early ensembling",
    "late_ensembling_min2.png": "Late ensembling\n(>=2 prompts)",
    "late_ensembling_min10.png": "Late ensembling\n(>=10 prompts)",
}

# Filter out "gt.png"
map_titles_filtered = {k: v for k, v in map_titles.items() if k != "gt.png"}

for filename, title in map_titles_filtered.items():
    map_path = os.path.join(folder_path, filename)
    img = Image.open(map_path)
    cropped_img = img.crop((int(img.width * 0.2), 0, int(img.width * 0.7), img.height))
    maps.append((cropped_img, title))

# Calculate the width and height of the individual cropped maps
first_map_path = os.path.join(folder_path, "zero_shot_contrastive_25.png")
first_map_img = Image.open(first_map_path)
original_map_width, original_map_height = first_map_img.size

# Calculate the width and height of the individual cropped maps
total_width = num_cols * (1 - 0.2 * 2) * original_map_width
total_height = num_rows * original_map_height + legend_height

# Create a blank canvas for the collage
collage = Image.new("RGB", (int(total_width), int(total_height)), (255, 255, 255))

# Create subplots with titles
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
fig.suptitle("Scenicness predictions", fontsize=20, fontweight="bold")

# Plot the maps on the subplots
for i, (map_img, title) in enumerate(maps):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].imshow(map_img)
    axs[row, col].set_title(f"{title}", fontweight="bold")
    axs[row, col].axis("off")

# Hide empty subplots
for i in range(len(maps), num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].axis("off")

# Get the height of the header
header_height = axs[0, 0].get_position().get_points()[1, 1]

# Add legend with spectral color ramp
legend_ax = fig.add_axes([0.375, header_height + 0.02, 0.25, 0.025])
cmap = cm.Spectral
norm = plt.Normalize(vmin=1, vmax=10)
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=legend_ax, orientation="horizontal")
cbar.set_ticks(range(1, 11))

plt.subplots_adjust(top=0.8)  # Increase or decrease the top value to adjust the spacing

# Save the collage
plt.savefig(output_path, dpi=300)
plt.close()
