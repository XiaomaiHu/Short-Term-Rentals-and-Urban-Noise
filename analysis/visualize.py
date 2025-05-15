# Import library
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from joypy import joyplot

from shapely.geometry import box
from shapely import wkt

import plotly.express as px

from scipy.stats import ttest_rel

#############################Descriptive Statistics###################################

######################################################################
#################### Plot 1: Room Type Distribution ##################
######################################################################

def plot_room_type_distribution(df, title, filename=None):
    # Set the figure size
    plt.figure(figsize=(9, 7))

    # Define custom colors for bars
    colors = ["#ffb482","#8de5a1", "#d0bbff"]
    # Create the countplot grouped by 'room_type'
    ax = sns.countplot(data=df, x="room_type", hue="room_type")

    # Remove the top and right spines (border lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add count labels inside each bar (in black)
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height * 0.6),  # Position inside bar
                ha='center',
                va='center',
                fontsize=12,
                color='black')

    # Add title and axis labels
    plt.title(title, fontsize=15)
    plt.xlabel("Room Type", fontsize=12)
    plt.ylabel("Listing Count", fontsize=12)

    # Axis tick styling
    plt.xticks(rotation=10, fontsize=11)
    plt.yticks(fontsize=11)

    # Clean layout
    plt.tight_layout()

    # Save the figure
    if filename:
        vis_dir = os.path.join(os.path.dirname(__file__), "..", "vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, filename)
        plt.savefig(save_path, dpi=300)


    # Show the final chart
    plt.show()

################################################################################
############ Plot 2: Room Type Distribution for Short-Term Listings ############
################################################################################

def plot_short_term_room_type_distribution(df, title, filename=None):
    # Set the figure size
    plt.figure(figsize=(9, 7))

    # Define custom colors for bars
    colors = ["#ffb482", "#d0bbff"]

    # Create the countplot grouped by 'room_type'
    ax = sns.countplot(
        data=df,
        x="room_type",
        hue="room_type",
    )

    # Remove the top and right spines (border lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add count labels inside each bar (in black)
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height * 0.6),  # Position inside bar
                ha='center',
                va='center',
                fontsize=12,
                color='black'  # Black font for contrast inside light bars
            )

    # Add title and axis labels
    plt.title(title, fontsize=15)
    plt.xlabel("Room Type", fontsize=12)
    plt.ylabel("Listing Count", fontsize=12)

    # Axis tick styling
    plt.xticks(rotation=10, fontsize=11)
    plt.yticks(fontsize=11)

    # Clean layout
    plt.tight_layout()

    # Save the figure
    if filename:
        vis_dir = os.path.join(os.path.dirname(__file__), "..", "vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, filename)
        plt.savefig(save_path, dpi=300)

    # Show the final chart
    plt.show()

############################# Study Area Choose ###################################

########################################################################
####################### Plot1: Choropleth Map ##########################
############## Airbnb Distribution of NYC by Zipcode ###################
########################################################################

def plot_airbnb_distribution_by_zipcode(gdf, title, filename=None):
    fig, ax = plt.subplots(figsize=(6, 8))
    gdf.plot(column="counts", cmap="Reds", legend=True, linewidth=0.8, edgecolor="grey", ax=ax)
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")

    if filename:
        vis_dir = os.path.join(os.path.dirname(__file__), "..", "vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, filename)
        plt.savefig(save_path, dpi=300)

    plt.show()

##############################################################################
############################ Plot2: KDE Plot #################################
########################### Minimum Night Changes ############################
####################### Williamsburg vs. Brooklyn ############################
###################### Lower East Side vs. Manhattan #########################
##############################################################################

def plot_kde_min_nights_change(df1, df2, label1, label2, title, filename=None):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", context="talk")

    # Plot KDE curves
    ax = sns.kdeplot(df1["change_in_minimum_nights"], label=label1,
                     linewidth=2.5, color="royalblue", bw_adjust=1)
    ax = sns.kdeplot(df2["change_in_minimum_nights"], label=label2,
                     linewidth=2.5, color="darkorange", bw_adjust=1)

    # Extract the peak points for annotation
    x1, y1 = ax.lines[0].get_data()
    x2, y2 = ax.lines[1].get_data()
    peak1 = (x1[np.argmax(y1)], np.max(y1))
    peak2 = (x2[np.argmax(y2)], np.max(y2))

    # Add annotations with dashed arrows pointing to the peaks
    plt.annotate('Peak', xy=peak1, xytext=(peak1[0] - 20, peak1[1]),
                 arrowprops=dict(arrowstyle='->', lw=0.5, color='royalblue', linestyle='dashed'),
                 fontsize=12, color='blue', ha='left', va='center')

    plt.annotate('Peak', xy=peak2, xytext=(peak2[0] - 20, peak2[1]),
                 arrowprops=dict(arrowstyle='->', lw=0.5, color='darkorange', linestyle='dashed'),
                 fontsize=12, color='#E66100', ha='left', va='center')

    # Dashed grid lines for better readability
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)

    # Title and axis formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Change in Minimum Required Stay (Days)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Neighborhood", fontsize=12, title_fontsize=13,
               loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if filename:
        vis_dir = os.path.join(os.path.dirname(__file__), "..", "vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, filename)
        plt.savefig(save_path, dpi=300)

    plt.show()

################################# Exploratory Data Analysis  ################################

#############################################################################################
########################### Categorical Point Distribution Map ##############################
##################### Airbnb short-term rental distribution change ##########################
#############################################################################################

def plot_airbnb_change_distribution(gdf, boundary_gdf, street_gdf, title, filename = None):
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot the street segment data
    street_gdf.plot(ax=ax, linewidth=1, color="grey", alpha=0.2)

    # Plot the boundary
    boundary_gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.1)

    # Plot rental distribution categorized by change type
    gdf[gdf["rental_change_type"] == "Longer Rental"].plot(ax=ax, color="blue", markersize=5, label="Longer Rental")
    gdf[gdf["rental_change_type"] == "No Change"].plot(ax=ax, color="red", markersize=5, label="No Change")
    gdf[gdf["rental_change_type"] == "Shorter Rental"].plot(ax=ax, color="yellow", markersize=5, label="Shorter Rental")

    # Remove grid lines
    plt.grid(False)

    # Title and labels
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    # Move legend outside the plot
    plt.legend(title="Change Type", bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.tight_layout()

    if filename:
        vis_dir = os.path.join(os.path.dirname(__file__), "..", "vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, filename)
        plt.savefig(save_path, dpi=300)

    plt.show()

#############################Descriptive Statistics###################################

######################################################################
#################### Plot 1: noise complaint count ##################
######################################################################
def plot_noise_count_comparison(pre_df, post_df, title):
    # Count complaints
    pre_counts = pre_df['Descriptor'].value_counts().reset_index()
    pre_counts.columns = ['Descriptor', 'Count']
    pre_counts['Period'] = 'Pre-policy:2022.09-2023.08'

    post_counts = post_df['Descriptor'].value_counts().reset_index()
    post_counts.columns = ['Descriptor', 'Count']
    post_counts['Period'] = 'Post-policy:2023.09-2024.08'

    # Combine
    combined = pd.concat([pre_counts, post_counts])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=combined, x= 'Descriptor', y='Count', hue='Period', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Noise Type")
    plt.tight_layout()
    plt.grid(False)

    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)
    print(f" '{title}'  generated")

    plt.show()

################################# Exploratory Data Analysis  ################################

######################################################################
######### Plot 2: Noise Complain Point Distribution ##################
######################################################################
def plot_noise_spatial_distribution(gdf, type, street_gdf, boundary_gdf, title):

    n = len(type)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))
    axes = axes.flatten()

    colors = cm.get_cmap('viridis_r', n + 1)

    for i, descriptor in enumerate(type):
        ax = axes[i]
        subset = gdf[gdf['Descriptor'] == descriptor]
        subset.plot(ax=ax, markersize=1, alpha=1, color=colors(i))
        street_gdf.plot(ax=ax, linewidth=1, color="grey", alpha=0.2)
        boundary_gdf.plot(ax=ax, edgecolor="black", facecolor="none", alpha=0.5)
        ax.set_title(descriptor)
        ax.axis("off")

    # Turn off extra subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()


    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)
    print(f" '{title}'  generated")

    plt.show()

######################################################################
############## Plot 3: Noise Complain KDE Estimation  ################
######################################################################
def plot_noise_kde_distribution(gdf, type, street_gdf, boundary_gdf, title):

    n = len(type)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5))
    axes = axes.flatten()

    for i, descriptor in enumerate(type):
        ax = axes[i]
        subset = gdf[gdf['Descriptor'] == descriptor]

        # Plot base map
        street_gdf.plot(ax=ax, linewidth=1, color="grey", alpha=0.2)
        boundary_gdf.plot(ax=ax, edgecolor="black", facecolor="none", alpha=0.5)

        # Extract coordinates and apply KDE
        x = subset.geometry.x
        y = subset.geometry.y
        sns.kdeplot(x=x, y=y, fill=True, cmap="viridis", ax=ax, bw_adjust=1, levels=10, alpha=0.9)

        ax.set_title(descriptor)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()


    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)
    print(f" '{title}'  generated")

    plt.show()

######################################################################
########### Plot 4: Noise Complaint Hourly Distribution  #############
######################################################################
def plot_noise_joyplot_by_hour(gdf, title):
    plt.figure(figsize=(16, 6))

    # joyplot 返回 fig, axes
    fig, axes = joyplot(
        data=gdf,
        by='Descriptor',
        column='hour',
        range_style='own',
        bins=24,
        x_range=(0, 23),
        grid=True,
        linewidth=1.2,
        overlap=0.5,
        colormap=cm.get_cmap('viridis_r')
    )

    plt.title(title, fontsize=8)

    axes[-1].set_xlabel("Hour of Day", fontsize=8)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.xticks(ticks=[3, 6, 9, 12, 15, 18, 21, 24],
               labels=["3", "6", "9", "12", "15", "18", "21", "24"],
               fontsize=8)

    plt.tight_layout()

    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)
    print(f" '{title}' generated")

    plt.show()

######################################################################
########### Plot 5: Noise Complaint Monthly Distribution  ############
######################################################################
def plot_monthly_trend_by_type(gdf, title, ybar):

    # Fixed policy month
    policy_month = '2023-09'

    # Group and pivot
    monthly_noise = gdf.groupby(['month_fmt', 'Descriptor']).size().reset_index(name='count')
    monthly_noise = monthly_noise.sort_values('month_fmt')
    pivot_df = monthly_noise.pivot_table(index='month_fmt', columns='Descriptor', values='count', fill_value=0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df.index = pivot_df.index.astype(str)

    # Plot
    plt.figure(figsize=(12, 8))
    pivot_df.plot(ax=plt.gca(), colormap=cm.get_cmap('viridis_r'), marker='o', markersize=4)
    plt.title(title, fontsize=12)
    plt.xlabel("Month")
    plt.ylabel("Number of Complaints")
    plt.xticks(rotation=45)
    plt.ylim(0, ybar)
    plt.grid(False)

    # Policy line
    if policy_month in pivot_df.index:
        policy_idx = pivot_df.index.get_loc(policy_month)
        plt.axvline(x=policy_idx, color='grey', linestyle='--', linewidth=1.5)
        plt.text(policy_idx + 0.2, plt.ylim()[1]*0.9, 'Policy Implemented', color='black')

    plt.legend(title="Descriptor")
    plt.tight_layout()

    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)
    print(f" '{title}'  generated")

    plt.show()

######################################################################
############# Plot 6: Noise Complaint Weekly Camparison  #############
######################################################################
def plot_weekly_ttest_comparison(pre_df, post_df, complaint):

    # Sort by week and reset index
    pre_sorted = pre_df.sort_values('week').reset_index(drop=True)
    post_sorted = post_df.sort_values('week').reset_index(drop=True)

    # Extract weekly columns
    col_pre = pre_sorted[complaint]
    col_post = post_sorted[complaint]

    # Paired t-test
    t_stat, p_val = ttest_rel(col_pre, col_post)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(col_pre.index, col_pre, label='Pre Policy', marker='o')
    plt.plot(col_post.index, col_post, label='Post Policy', marker='o')

    title = f"Residential{complaint} Complaints in Manhattan Study Area Weekly Comparison (p = {p_val:.3f})"
    plt.title(title, fontsize=14)
    plt.xlabel('Week Index')
    plt.ylabel('Weekly Complaint Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    os.makedirs("vis", exist_ok=True)
    filename = title.replace("/", "&") + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)

    print(f" '{title}'  generated")

    plt.show()

######################################################################
####################### Plot 7: Fishnes Plot  ########################
######################################################################
def create_fishnet(boundary_gdf, cell_size=500):
    xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
    rows = int(np.ceil((ymax - ymin) / cell_size))
    cols = int(np.ceil((xmax - xmin) / cell_size))

    polygons = []
    for i in range(cols):
        for j in range(rows):
            x0 = xmin + i * cell_size
            y0 = ymin + j * cell_size
            polygons.append(box(x0, y0, x0 + cell_size, y0 + cell_size))

    grid = gpd.GeoDataFrame(geometry=polygons, crs=boundary_gdf.crs)
    grid = gpd.overlay(grid, boundary_gdf, how='intersection')
    return grid

def plot_spatial_change_comparison(pre_gdf, post_gdf, airbnb_gdf, boundary_gdf, title):
    """
    Create fishnet and plot spatial distribution of:
    - Change in complaints (pre vs post)
    - Change in Airbnb rentals (e.g. 'Longer Rental')

    Parameters:
        pre_gdf (GeoDataFrame): Pre-policy filtered complaints of one type (e.g. Loud Music).
        post_gdf (GeoDataFrame): Post-policy filtered complaints of one type.
        airbnb_gdf (GeoDataFrame): Airbnb data with column 'rental_change_type' == 'Longer Rental'.
        boundary_gdf (GeoDataFrame): Study area boundary (e.g. zipcode).
        title_prefix (str): Will be added to titles and output filename.
    """
    # Generate fishnet grid
    fishnet = create_fishnet(boundary_gdf, cell_size=500)

    # Count pre-policy complaints
    joined = gpd.sjoin(pre_gdf, fishnet, how="left", predicate="within")
    fishnet["pre-count"] = 0
    pre_counts = joined.groupby("index_right").size()
    fishnet.loc[pre_counts.index, "pre-count"] = pre_counts.values

    # Count post-policy complaints
    joined = gpd.sjoin(post_gdf, fishnet, how="left", predicate="within")
    fishnet["post-count"] = 0
    post_counts = joined.groupby("index_right").size()
    fishnet.loc[post_counts.index, "post-count"] = post_counts.values

    # Count changed Airbnb
    joined = gpd.sjoin(airbnb_gdf, fishnet, how="left", predicate="within")
    fishnet["changed_airbnb"] = 0
    airbnb_counts = joined.groupby("index_right").size()
    fishnet.loc[airbnb_counts.index, "changed_airbnb"] = airbnb_counts.values

    # Calculate difference
    fishnet["changed_noise"] = fishnet["pre-count"] - fishnet["post-count"]

    # Plot side-by-side maps
    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

    fishnet.plot(
        column="changed_noise",
        cmap="viridis",
        linewidth=0.5,
        ax=axes[0],
        legend=True,
        vmin=-50, vmax=50,
        legend_kwds={'shrink': 0.6}
    )
    boundary_gdf.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
    axes[0].set_title(f"Loud Music/Party Noise Complaint Count Difference")
    axes[0].axis("off")

    fishnet.plot(
        column="changed_airbnb",
        cmap="viridis",
        linewidth=0.5,
        ax=axes[1],
        legend=True,
        legend_kwds={'shrink': 0.6}
    )
    boundary_gdf.boundary.plot(ax=axes[1], color="black", linewidth=0.5)
    axes[1].set_title(f" Changed Airbnb Short Listing Count")
    axes[1].axis("off")

    plt.tight_layout()

    # Save
    os.makedirs("vis", exist_ok=True)
    filename = title + ".png"
    filepath = os.path.join("vis", filename)
    plt.savefig(filepath, dpi=300)

    print(f" '{title}' generated")
    plt.show()

