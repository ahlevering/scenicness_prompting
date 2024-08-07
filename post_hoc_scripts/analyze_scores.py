import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

ensemble_file = "data/outputs/ensembling/ensembling_preds_clip.gpkg"
ensemble_results = gpd.read_file(ensemble_file)

corine_file = "data/son_pts_aux_info.geojson"
corine_pts = gpd.read_file(corine_file)

corine_pts = corine_pts[["ID", "Average", "lc"]]
corine_pts["lc"] = corine_pts["lc"].str[:2]
corine_pts = corine_pts.rename({'Average_x': 'Average'})

# Merge the dataframes
corine_pts = corine_pts.merge(ensemble_results, on="ID")
# Get the unique LC classes and sort them
lc_classes = sorted(corine_pts["lc"].unique())

# Define the color palette based on the first digit of the LC class
color_dict = {
    "1": "#808080",  # Gray
    "2": "#FFFF00",  # Yellow
    "3": "#006400",  # Dark Green
    "4": "#40E0D0",  # Turquoise
    "5": "#00008B",  # Dark Blue
}

legend_dict = {
    "#808080": "Artificial surfaces",
    "#FFFF00": "Agricultural areas",
    "#006400": "Forests and seminatural areas",
    "#40E0D0": "Wetlands",
    "#00008B": "Water bodies",
}

# Assign colors based on the first digit of the LC class
colors = [color_dict[lc[0]] for lc in lc_classes]

# Modify the LC classes to be separated by a dot
lc_classes_modified = [f"{lc[0]}.{lc[1]}" for lc in lc_classes]

# Define the prediction columns
prediction_columns = ["Average", "preds_2"]

# Create the box plot using Seaborn
sns.set_style("whitegrid")
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 3))  # Adjusted figsize for shorter plot

# Iterate over each prediction column and create a separate box plot
for i, pred_column in enumerate(prediction_columns):
    ax = axs[i]

    # Sort the dataframe based on LC classes for correct x-axis ordering
    corine_pts_sorted = corine_pts.sort_values("lc", key=lambda x: x.map({lc: i for i, lc in enumerate(lc_classes)}))

    # Plot box plots for each LC class with assigned colors
    sns.boxplot(x=corine_pts_sorted["lc"], y=corine_pts_sorted[pred_column], palette=colors, ax=ax, showfliers=False)

    # Customize the plot
    ax.set_ylabel("Scenicness", fontsize=14)
    if pred_column == "Average":
        ax.set_title("SON Ratings", fontweight="bold", fontsize=16, pad=10)
    elif pred_column == "preds_2":
        ax.set_title("Late Ensembling - 2 or More Prompts", fontweight="bold", fontsize=16, pad=10)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticklabels(lc_classes_modified)  # Use modified LC classes

# Create a legend for the assigned colors
lc1_classes = sorted(set([lc[0] for lc in lc_classes]))
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[lc]) for lc in lc1_classes]
legend_labels = [legend_dict[color_dict[lc[0]]] for lc in lc1_classes]
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(colors),
    title="CORINE Level-1 class",
    prop={"size": 12},
)
legend.get_title().set_fontsize(12)  # Set legend title font size
legend.get_title().set_fontweight("bold")  # Set legend title font weight

for ax in axs:
    ax.set_xlabel("")

fig.tight_layout()

# Adjust the bottom margin of the figure to add space for the legend
plt.subplots_adjust(bottom=0.2)

plt.savefig("lc_class_box_plots.png", bbox_inches="tight", dpi=300)
plt.close()
