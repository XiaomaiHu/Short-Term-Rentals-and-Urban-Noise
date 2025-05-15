# Import library
import pandas as pd
import os
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


import numpy as np
import plotly.express as px

from shapely.geometry import Point
from shapely.geometry import Point
from shapely.wkt import loads

########################################################################
############## Part 1: Process and visualize Airbnb data ###############
########################################################################
# Import functions
from process_data import process_airbnb_data
merged_df, final_merged_df, Manhattan_merged_df, Manhattan_final_merged_df = process_airbnb_data()

from visualize import (
    plot_room_type_distribution,
    plot_short_term_room_type_distribution,
    plot_kde_min_nights_change,
    plot_airbnb_distribution_by_zipcode,
    plot_airbnb_change_distribution
)

# After processing, load cleaned datasets
final_merged_df = pd.read_csv("../data/processed/Cleaned_data_11211.csv")
Manhattan_final_merged_df = pd.read_csv("../data/processed/Cleaned_data_10002.csv")


########################################################################
############ Describe statistics: Brooklyn (All listings) ##############
########################################################################

# Note:
# (i) "merged_df" refers to all listings in Brooklyn
# (ii) "final_merged_df" refers to listings only in Williamsburg (ZIP: 11211)


# Step 1: merge_df refering the whole brooklyn area
merged_df["change_in_minimum_nights"] = merged_df["minimum_nights_y"] - merged_df["minimum_nights_x"]

# Step 2: Filter short-term rentals
b_short_rentals_2023 = merged_df[merged_df["minimum_nights_x"] < 30].copy()

# Step 3: Categorize change type
b_short_rentals_2023["rental_change_type"] = b_short_rentals_2023["change_in_minimum_nights"].apply(
    lambda x: "Longer Rental" if x > 0 else ("Shorter Rental" if x < 0 else "No Change")
)

# Step 4: Apply IQR filtering within each category
def remove_outliers_by_iqr(group):
    Q1 = group["change_in_minimum_nights"].quantile(0.25)
    Q3 = group["change_in_minimum_nights"].quantile(0.75)
    IQR = Q3 - Q1
    return group[
        (group["change_in_minimum_nights"] >= Q1 - 1.5 * IQR) &
        (group["change_in_minimum_nights"] <= Q3 + 1.5 * IQR)
    ]

# Apply function group-wise
b_short_rentals_2023 = b_short_rentals_2023.groupby(
    "rental_change_type", group_keys=False
).apply(remove_outliers_by_iqr, include_groups=False)


#whole brooklyn days change
b_short_rentals_2023['change_in_minimum_nights'].describe()
print(b_short_rentals_2023['change_in_minimum_nights'].describe())


#######################################################################################
##################### Analysis: Williamsburg (ZIP 11211) ##############################
#######################################################################################

#############################Descriptive Statistics###################################

# Step 1: Calculate change in minimum stay (in days)
# This subtracts the previous value (minimum_nights_x) from the latest (minimum_nights_y)
# Positive result = now requires longer stay; negative = shorter stay
final_merged_df.loc[:, "change_in_minimum_nights"] = (
    final_merged_df["minimum_nights_y"] - final_merged_df["minimum_nights_x"]
)
# Step 2: Filter to only include listings that were originally short-term (minimum stay < 30 days)
# We focus on listings that are potentially affected by short-term rental regulations
w_short_rentals_2023 = final_merged_df[final_merged_df["minimum_nights_x"] < 30].copy()

# Step 3: Categorize listings based on how their minimum stay changed
# Longer Rental  ->increased minimum stay
# Shorter Rental -> decreased minimum stay
# No Change      ->  no difference
w_short_rentals_2023["rental_change_type"] = w_short_rentals_2023["change_in_minimum_nights"].apply(
    lambda x: "Longer Rental" if x > 0 else ("Shorter Rental" if x < 0 else "No Change")
)

# Step 4: Remove outliers within each change category using the IQR method
# This ensures that extreme changes (e.g. mistakenly entered or out-of-policy listings) are excluded

w_short_rentals_2023 = w_short_rentals_2023.groupby(
    "rental_change_type", group_keys=False
).apply(remove_outliers_by_iqr,include_groups=False)

# Step 5: Generate descriptive statistics for Williamsburg short-term listings
# Helps summarize how much the minimum stay requirements have shifted post-policy
w_short_rentals_2023['change_in_minimum_nights'].describe()
w_short_rentals_2023["room_type"].unique()

################################################################################
############# Plot 1: Room Type Distribution in Brooklyn #######################
########################## Study Area(Williamsburg)#############################
################################################################################

plot_room_type_distribution(
    df=final_merged_df,
    title="Distribution of Room Types in Brooklyn Study Area (Williamsburg)",
    filename="distribution_room_types_brooklyn_williamsburg.png"
)


################################################################################
########################### Williamsburg Room Type##############################
############ Plot 2: Room Type Distribution for Short-Term Listings ############
################################################################################

plot_short_term_room_type_distribution(
    df=w_short_rentals_2023,
    title="Distribution of Short-Term Listings (<30 Days) by Room Type in Brooklyn Study Area(Williamsburg)",
    filename="Distribution of Short-Term Listings (<30 Days) by Room Type in Brooklyn Study Area(Williamsburg).png"
)

############# Price Difference Analysis #############

#calculate the whole brooklyn price difference
#whole brooklyn days change
b_short_rentals_2023['price_difference'] = b_short_rentals_2023['price_y'] - b_short_rentals_2023['price_x']

#11211
w_short_rentals_2023.loc[:,'price_difference'] = w_short_rentals_2023['price_y'] - w_short_rentals_2023['price_x']

#drop na
w_short_rentals_2023[w_short_rentals_2023['price_difference'].isna()]
w_short_rentals_2023 = w_short_rentals_2023.dropna(subset=['price_difference'])
w_short_rentals_2023.columns
#Outlier Detection (Biggest Price Changes)
#See which listings had unusual price jumps:
w_short_rentals_2023.sort_values

top10_days_change = w_short_rentals_2023.sort_values(by='change_in_minimum_nights', ascending=False)
top10_days_change = top10_days_change[['id', 'room_type', 'minimum_nights_x', 'minimum_nights_y', 'change_in_minimum_nights', 'price_difference']]
top10_days_change.head(10)

############################# Study Area Choose ###################################

########################################################################
############# Choropleth Map: Airbnb Listings by Zipcode ###############
########################################################################

# Load the zipcode boundary data
zipcode_boundary = gpd.read_file("data/raw/zipcode_geometry.geojson")
zipcode_boundary = zipcode_boundary.to_crs(epsg=2263)  # Convert CRS

manhattan_brooklyn_zipcodes = [
    11201, 11203, 11204, 11205, 11206, 11207, 11208, 11209, 11210,
    11211, 11212, 11213, 11214, 11215, 11216, 11217, 11218, 11219,
    11220, 11221, 11222, 11223, 11224, 11225, 11226, 11228, 11229,
    11230, 11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238,
    11239, 11241, 11243, 11249, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10009, 10010,
    10011, 10012, 10013, 10014, 10016, 10017, 10018, 10019, 10021,
    10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030,
    10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039,
    10040, 10044, 10069, 10103, 10119, 10128, 10162, 10165, 10170,
    10173, 10199, 10279, 10280, 10282
]

# Ensure zip codes are strings to match GeoDataFrame column type
manhattan_brooklyn_zipcodes = [str(z) for z in manhattan_brooklyn_zipcodes]
zipcode_boundary["modzcta"] = zipcode_boundary["modzcta"].astype(str)

zipcode_boundary = zipcode_boundary[zipcode_boundary["modzcta"].isin(manhattan_brooklyn_zipcodes)]

# Load the Airbnb data
airbnb_data = pd.read_csv("data/raw/NYC-Airbnb-2024.csv", low_memory=False)
airbnb_data = airbnb_data.dropna(subset=["latitude", "longitude"])

# Convert Airbnb data to GeoDataFrame
gdf_airbnb = gpd.GeoDataFrame(
    airbnb_data, geometry=gpd.points_from_xy(airbnb_data["longitude"], airbnb_data["latitude"]), crs="EPSG:4326"
)
gdf_airbnb = gdf_airbnb.to_crs(epsg=2263)

# Perform spatial join to associate each Airbnb listing with a zipcode
airbnb_with_zipcode = gpd.sjoin(gdf_airbnb, zipcode_boundary, how="left", predicate="within")

# Aggregating Airbnb data by zipcode
zipcode_airbnb_counts = airbnb_with_zipcode.groupby(["modzcta"]).size().reset_index(name="counts")
print(zipcode_airbnb_counts.head())


# Get the top 5 zip codes with the highest Airbnb counts
top_5_zipcodes = zipcode_airbnb_counts.nlargest(5, 'counts')

# Print the result
print("Top 5 zip codes with the highest Airbnb counts:")
print(top_5_zipcodes)

# Check zipcode value type
zipcode_boundary["modzcta"] = zipcode_boundary["modzcta"].astype("int64")
zipcode_airbnb_counts["modzcta"] = zipcode_airbnb_counts["modzcta"].astype("int64")
print(zipcode_boundary["modzcta"].dtype)
print(zipcode_airbnb_counts["modzcta"].dtype)

# Merge two DataFrames
zipcode_airbnb_gdf = zipcode_boundary.merge(
    zipcode_airbnb_counts[["modzcta", "counts"]],
    left_on="modzcta",
    right_on="modzcta",
    how="left"
)
# Fill NaN counts with zero
zipcode_airbnb_gdf["counts"] = zipcode_airbnb_gdf["counts"].fillna(0)

########################################################################
####################### Plot1: Choropleth Map ##########################
############## Airbnb Distribution of NYC by Zipcode ###################
########################################################################

plot_airbnb_distribution_by_zipcode(
    gdf=zipcode_airbnb_gdf,
    title="Airbnb Distribution of NYC by Zipcode",
    filename="Airbnb Distribution of NYC by Zipcode.png"
)

########################################################################

##############################################################################
############################ Plot2: KDE Plot #################################
########################### Minimum Night Changes ############################
####################### Williamsburg vs. Brooklyn ############################
##############################################################################

# Brooklyn
plot_kde_min_nights_change(
    df1=w_short_rentals_2023,
    df2=b_short_rentals_2023,
    label1="Williamsburg",
    label2="Brooklyn",
    title="KDE of Change in Minimum Nights in Brooklyn",
    filename="KDE of Change in Minimum Nights in Brooklyn.png"
)

##############################################################################

##########################################################################
############# Summary Chart: Rental Change Type Distribution #############
##########################################################################

# Step 1: Filter for short-term rentals in 2023 (minimum_nights_x < 30)
w_short_rentals_2023 = final_merged_df[final_merged_df["minimum_nights_x"] < 30].copy()
# Step 2: Calculate the difference in minimum nights between 2023 and 2024
w_short_rentals_2023["change_in_minimum_nights"] = (
      w_short_rentals_2023["minimum_nights_y"] - w_short_rentals_2023["minimum_nights_x"]
    )

# Step 3: Categorize changes as 'Longer Rental', 'Shorter Rental', or 'No Change'
w_short_rentals_2023["rental_change_type"] = w_short_rentals_2023["change_in_minimum_nights"].apply(
      lambda x: "Longer Rental" if x > 0 else ("Shorter Rental" if x < 0 else "No Change")
    )

# Step 4: Count and calculate the ratio of each rental change category
change_counts = w_short_rentals_2023["rental_change_type"].value_counts()
change_ratios = change_counts / change_counts.sum()

# Step 5: Print count and ratio for each rental change type
print("Rental Change Type Counts:")
print(change_counts)
print("\nRental Change Type Ratios:")
print(change_ratios)

############# Map: Airbnb Short-Term Rental Change (Street-Level) #############

# Step 1: Load the zipcode boundary data
zipcode_boundary = gpd.read_file("data/raw/new_boundary.geojson")
zipcode_boundary = zipcode_boundary.to_crs(epsg=4326)  # Convert CRS

# Step 2: Convert short-term rental data to a GeoDataFrame
gdf = gpd.GeoDataFrame(
    w_short_rentals_2023,
    geometry=gpd.points_from_xy(w_short_rentals_2023.longitude_x, w_short_rentals_2023.latitude_x),
    crs="EPSG:4326"
).to_crs(epsg=4326)

# Step 3: Load street segment data
street_gdf = gpd.read_file("data/raw/Street_0405.geojson")
street_gdf = street_gdf.to_crs(epsg=4326)

# Step 4: Create visualizations to explore Airbnb short-term rental distribution change in 11211 (Brooklyn) from 2023 to 2024
#############################################################################################
########################### Categorical Point Distribution Map ##############################
##################### Airbnb short-term rental distribution change ##########################
#############################################################################################

plot_airbnb_change_distribution(
    gdf=gdf,
    boundary_gdf=zipcode_boundary,
    street_gdf=street_gdf,
    title="Change in Airbnb Short-Term Rentals in the Williamsburg from 2023 to 2024",
    filename="Change in Airbnb Short-Term Rentals in the Williamsburg from 2023 to 2024.png"
)

#############################################################################################


#######################################################################################
##################### Analysis: Lower East Side (ZIP 11211) ###########################
#######################################################################################

############################### Descriptive statistics#################################

#note: (i)the Manhattan_merged_df is referring whole Manhattan data
 #(ii)add "final", "Manhattan_final_merged_df" is referring whole 10002 data

# Step 1: manhattan 2023 short rental list. here Manhattan_merged_df is indicate the whole dataset of manhattan
Manhattan_merged_df["change_in_minimum_nights"] = Manhattan_merged_df["minimum_nights_y"] - Manhattan_merged_df["minimum_nights_x"]

# Step 2: Filter short-term rentals (< 30 days in 2023 data)
Manhattan_b_short_rentals_2023 = Manhattan_merged_df[Manhattan_merged_df["minimum_nights_x"] < 30].copy()

# Step 3: Categorize change type
Manhattan_b_short_rentals_2023["rental_change_type"] = Manhattan_b_short_rentals_2023["change_in_minimum_nights"].apply(
    lambda x: "Longer Rental" if x > 0 else ("Shorter Rental" if x < 0 else "No Change")
)

# Step 4: Apply IQR filtering within each category
def remove_outliers_by_iqr(group):
    Q1 = group["change_in_minimum_nights"].quantile(0.25)
    Q3 = group["change_in_minimum_nights"].quantile(0.75)
    IQR = Q3 - Q1
    return group[
        (group["change_in_minimum_nights"] >= Q1 - 1.5 * IQR) &
        (group["change_in_minimum_nights"] <= Q3 + 1.5 * IQR)
    ]

# Apply function group-wise

Manhattan_b_short_rentals_2023 = Manhattan_b_short_rentals_2023.groupby(
    "rental_change_type", group_keys=False
).apply(remove_outliers_by_iqr, include_groups=False)

#whole brooklyn days change

##################### 100002 days changes analysis#####################
# Step 1: Calculate change in minimum stay (in days)
# This subtracts the previous value (minimum_nights_x) from the latest (minimum_nights_y)
# Positive result = now requires longer stay; negative = shorter stay
Manhattan_final_merged_df.loc[:, "change_in_minimum_nights"] = (
    Manhattan_final_merged_df["minimum_nights_y"] - Manhattan_final_merged_df["minimum_nights_x"]
)
# Step 2: Filter to only include listings that were originally short-term (minimum stay < 30 days)
# We focus on listings that are potentially affected by short-term rental regulations
china_town_short_rentals_2023 = Manhattan_final_merged_df[Manhattan_final_merged_df["minimum_nights_x"] < 30].copy()

# Step 3: Categorize listings based on how their minimum stay changed
# Longer Rental  ->increased minimum stay
# Shorter Rental -> decreased minimum stay
# No Change      ->  no difference
china_town_short_rentals_2023["rental_change_type"] = china_town_short_rentals_2023["change_in_minimum_nights"].apply(
    lambda x: "Longer Rental" if x > 0 else ("Shorter Rental" if x < 0 else "No Change"))


# Step 4: Remove outliers within each change category using the IQR method
# This ensures that extreme changes (e.g. mistakenly entered or out-of-policy listings) are excluded
china_town_short_rentals_2023 = china_town_short_rentals_2023.groupby(
    "rental_change_type", group_keys=False
).apply(remove_outliers_by_iqr, include_groups=False)


# Step 5: Generate descriptive statistics for Williamsburg short-term listings
china_town_short_rentals_2023['change_in_minimum_nights'].describe()

##############################################################################
############# Plot 1: Room Type Distribution in Manhattan ####################
########################## Study Area(China Town)#############################
##############################################################################

plot_room_type_distribution(
    df=Manhattan_final_merged_df,
    title="Distribution of Room Types in Manhattan Study Area(Lower East Side)",
    filename="Distribution of Room Types in Manhattan Study Area(Lower East Side).png"
)

################################################################################
#############################China Town Room Type###############################
############ Plot 2: Room Type Distribution for Short-Term Listings ############
################################################################################

plot_short_term_room_type_distribution(
    df=china_town_short_rentals_2023,
    title="Distribution of Short-Term Listings (<30 Days) by Room Type in Manhattan Study Area(Lower East Side)",
    filename="Distribution of Short-Term Listings (<30 Days) by Room Type in Manhattan Study Area(Lower East Side).png"
)

################################################################################

#calculate the whole brooklyn price difference
#whole brooklyn and 10002days change
Manhattan_b_short_rentals_2023['price_difference'] = Manhattan_b_short_rentals_2023['price_y'] - Manhattan_b_short_rentals_2023['price_x']

china_town_short_rentals_2023 .loc[:,'price_difference'] = china_town_short_rentals_2023['price_y'] - china_town_short_rentals_2023['price_x']

#drop na
china_town_short_rentals_2023[china_town_short_rentals_2023['price_difference'].isna()]
china_town_short_rentals_2023 = china_town_short_rentals_2023.dropna(subset=['price_difference'])
china_town_short_rentals_2023.columns

#Outlier Detection (Biggest Price Changes)
#See which listings had unusual price jumps:
china_town_short_rentals_2023.sort_values

china_town_top10_days_change = china_town_short_rentals_2023.sort_values(by='change_in_minimum_nights', ascending=False)
china_town_top10_days_change = china_town_top10_days_change[['id', 'room_type', 'minimum_nights_x', 'minimum_nights_y', 'change_in_minimum_nights', 'price_difference']]
china_town_top10_days_change.head(10)

##############################################################################
############################ Plot3: KDE Plot #################################
########################### Minimum Night Changes ############################
###################### Lower East Side vs. Manhattan #########################
##############################################################################

plot_kde_min_nights_change(
    df1=china_town_short_rentals_2023,
    df2=Manhattan_b_short_rentals_2023,
    label1="Lower East Side",
    label2="Manhattan",
    title="KDE of Change in Minimum Nights in Manhattan",
    filename="KDE of Change in Minimum Nights in Manhattan.png"
)

##############################################################################

###########################################################################
################################Spatial Graph##############################
###########################################################################

# Step 1: Load the zipcode boundary data
zipcode_boundary_10002 = gpd.read_file("data/raw/zipcodeboundary_10002.geojson")
zipcode_boundary_10002 = zipcode_boundary_10002.to_crs(epsg=4326)

# Step 2: Convert short-term rental data to a GeoDataFrame
gdf_10002 = gpd.GeoDataFrame(
    china_town_short_rentals_2023,
    geometry=gpd.points_from_xy(china_town_short_rentals_2023.longitude_x, china_town_short_rentals_2023.latitude_x),
    crs="EPSG:4326"
).to_crs(epsg=4326)

# Step 3: Load street segment data
street_gdf_10002 = gpd.read_file("data/raw/streetsegment_10002.geojson")
street_gdf_10002 = street_gdf_10002.to_crs(epsg=4326)


print("gdf shape:", gdf.shape)
print("gdf columns:", gdf.columns)
print("rental_change_type unique values:", gdf["rental_change_type"].unique())

# Step 4: Create visualizations to explore Airbnb short-term rental distribution change in 10002 (Manhattan) from 2023 to 2024
#############################################################################################
########################### Categorical Point Distribution Map ##############################
##################### Airbnb short-term rental distribution change ##########################
#############################################################################################
def classify_change(diff):
    if diff > 0:
        return "Longer Rental"
    elif diff < 0:
        return "Shorter Rental"
    else:
        return "No Change"

gdf_10002["rental_change_type"] = gdf_10002["change_in_minimum_nights"].apply(classify_change)


plot_airbnb_change_distribution(
    gdf=gdf_10002,
    boundary_gdf=zipcode_boundary_10002,
    street_gdf=street_gdf_10002,
    title="Change in Airbnb Short-Term Rentals in the Lower East Side from 2023 to 2024",
    filename="Change in Airbnb Short-Term Rentals in the Lower East Side from 2023 to 2024.png"
)


########################################################################
############## Part 2: Process and visualize Noise data ################
########################################################################
# Import functions
from process_data import process_noise_data

from visualize import (
    plot_noise_count_comparison,
    plot_noise_spatial_distribution,
    plot_noise_kde_distribution,
    plot_noise_joyplot_by_hour,
    plot_monthly_trend_by_type,
    plot_weekly_ttest_comparison,
    plot_spatial_change_comparison
)

gdf_noise_all_brooklyn, gdf_noise_all_manhattan, pre_policy_brooklyn, post_policy_brooklyn, pre_policy_manhattan, post_policy_manhattan = process_noise_data()

########################################################################
###################### Temporal Aggregation: Monthly ##################
########################################################################

monthly_noise_brooklyn = gdf_noise_all_brooklyn.groupby(['month_fmt', 'Descriptor']).size().reset_index(name='count')
monthly_noise_manhattan = gdf_noise_all_manhattan.groupby(['month_fmt', 'Descriptor']).size().reset_index(name='count')

########################################################################
###################### Temporal Aggregation: Hourly ###################
########################################################################

pre_hourly_noise_brookyln = pre_policy_brooklyn.groupby(['hour', 'Descriptor']).size().reset_index(name='count')
pre_hourly_noise_manhattan = pre_policy_manhattan.groupby(['hour', 'Descriptor']).size().reset_index(name='count')

post_hourly_noise_brookyln = post_policy_brooklyn.groupby(['hour', 'Descriptor']).size().reset_index(name='count')
post_hourly_noise_manhattan = post_policy_manhattan.groupby(['hour', 'Descriptor']).size().reset_index(name='count')

########################################################################
###################### Temporal Aggregation: Weekly ###################
########################################################################

# Step 1: Add 'week' column to enable weekly aggregation (week starts on Monday)
pre_policy_brooklyn['week'] = pre_policy_brooklyn['datetime_fmt'].dt.to_period('W-MON').apply(lambda r: r.start_time)
post_policy_brooklyn['week'] = post_policy_brooklyn['datetime_fmt'].dt.to_period('W-MON').apply(lambda r: r.start_time)
pre_policy_manhattan['week'] = pre_policy_manhattan['datetime_fmt'].dt.to_period('W-MON').apply(lambda r: r.start_time)
post_policy_manhattan['week'] = post_policy_manhattan['datetime_fmt'].dt.to_period('W-MON').apply(lambda r: r.start_time)

# Step 2: Calculate weekly complaint counts by descriptor
# Brooklyn
pre_weekly_count_brooklyn = pd.pivot_table(pre_policy_brooklyn, index='week', columns='Descriptor', aggfunc='size', fill_value=0)
post_weekly_count_brooklyn = pd.pivot_table(post_policy_brooklyn, index='week', columns='Descriptor', aggfunc='size', fill_value=0)

# Add total weekly counts
pre_weekly_count_brooklyn['Total'] = pre_weekly_count_brooklyn.sum(axis=1)
post_weekly_count_brooklyn['Total'] = post_weekly_count_brooklyn.sum(axis=1)

# Reset index for plotting
pre_weekly_count_brooklyn = pre_weekly_count_brooklyn.reset_index()
post_weekly_count_brooklyn = post_weekly_count_brooklyn.reset_index()

# Manhattan
pre_weekly_count_manhattan = pd.pivot_table(pre_policy_manhattan, index='week', columns='Descriptor', aggfunc='size', fill_value=0)
post_weekly_count_manhattan = pd.pivot_table(post_policy_manhattan, index='week', columns='Descriptor', aggfunc='size', fill_value=0)

# Add total weekly counts
pre_weekly_count_manhattan['Total'] = pre_weekly_count_manhattan.sum(axis=1)
post_weekly_count_manhattan['Total'] = post_weekly_count_manhattan.sum(axis=1)

# Reset index for plotting
pre_weekly_count_manhattan = pre_weekly_count_manhattan.reset_index()
post_weekly_count_manhattan = post_weekly_count_manhattan.reset_index()


################################################################################
########### Plot 1: Pre vs Post Policy Noise Complaint Counts ##################
################################################################################
plot_noise_count_comparison(
    pre_df=pre_policy_manhattan,
    post_df=post_policy_manhattan,
    title="Pre- and Post-policy Residential Noise Complaint Counts in Manhattan Study Area"
)

plot_noise_count_comparison(
    pre_df=pre_policy_brooklyn,
    post_df=post_policy_brooklyn,
    title="Pre- and Post-policy Residential Noise Complaint Counts in Brooklyn Study Area"
)

################################################################################
########### Plot 2: Spatial Distribution of Residential Noise  #################
################################################################################
############################## Manhattan Study Area ############################
# Get residential noise type
residential_descriptor = gdf_noise_all_manhattan['Descriptor'].unique()

plot_noise_spatial_distribution(
    gdf=gdf_noise_all_manhattan,
    type=residential_descriptor,
    street_gdf=street_gdf_10002.to_crs(epsg=2263),
    boundary_gdf=zipcode_boundary_10002.to_crs(epsg=2263),
    title="Residential Noise Complaints Spatial Distribution in Manhattan Study Area"
)

############################### Brooklyn Study Area ############################
plot_noise_spatial_distribution(
    gdf=gdf_noise_all_brooklyn,
    type=residential_descriptor,
    street_gdf=street_gdf.to_crs(epsg=2263),
    boundary_gdf=zipcode_boundary.to_crs(epsg=2263),
    title="Residential Noise Complaints Spatial Distribution in Brooklyn Study Area"
)

################################################################################
############### Plot 3: KDE Estimation of Residential Noise  ###################
################################################################################
############################## Manhattan Study Area ############################
plot_noise_kde_distribution(
    gdf=gdf_noise_all_manhattan,
    type=residential_descriptor,
    street_gdf=street_gdf_10002.to_crs(epsg=2263),
    boundary_gdf=zipcode_boundary_10002.to_crs(epsg=2263),
    title="Residential Noise Complaints KDE Estimation in Manhattan Study Area"
)

############################### Brooklyn Study Area ############################
plot_noise_kde_distribution(
    gdf=gdf_noise_all_brooklyn,
    type=residential_descriptor,
    street_gdf=street_gdf.to_crs(epsg=2263),
    boundary_gdf=zipcode_boundary.to_crs(epsg=2263),
    title="Residential Noise Complaints KDE Estimation in Brooklyn Study Area"
)

################################################################################
############# Plot 4: Hourly Joyplot Before Policy Implementation  ############
################################################################################
############################## Manhattan Study Area ############################
plot_noise_joyplot_by_hour(
    gdf= pre_hourly_noise_manhattan,
    title="Pre-policy Residential Noise Complaints Hourly Distribution in Manhattan Study Area"
)
plot_noise_joyplot_by_hour(
    gdf= post_hourly_noise_manhattan,
    title="Post-policy Residential Noise Complaints Hourly Distribution in Manhattan Study Area"
)
############################### Brooklyn Study Area ############################
plot_noise_joyplot_by_hour(
    gdf= pre_hourly_noise_brookyln ,
    title="Pre-policy Residential Noise Complaints Hourly Distribution in Brooklyn Study Area"
)
plot_noise_joyplot_by_hour(
    gdf= post_hourly_noise_brookyln ,
    title="Post-policy Residential Noise Complaints Hourly Distribution in Brooklyn Study Area"
)

################################################################################
########### Plot 5: Monthly Trend of Residential Noise by Descriptor############
################################################################################
############################## Manhattan Study Area ############################
plot_monthly_trend_by_type(
    gdf=gdf_noise_all_manhattan,
    title="Monthly Trend of Residential Noise Complaints by Type in Manhattan Study Area (Pre vs Post Policy)",
    ybar = 500
)

############################### Brooklyn Study Area ############################
plot_monthly_trend_by_type(
    gdf=gdf_noise_all_brooklyn,
    title="Monthly Trend of Residential Noise Complaints by Type in Brooklyn Study Area (Pre vs Post Policy)",
    ybar = 750
)

################################################################################
######### Plot 5: Weekly Trend of Loud Music/Party Complaints with t-test ######
################################################################################
############################## Manhattan Study Area ############################
plot_weekly_ttest_comparison(
    pre_df=pre_weekly_count_manhattan,
    post_df=post_weekly_count_manhattan,
    complaint="Loud Music/Party"
)

############################### Brooklyn Study Area ############################
plot_weekly_ttest_comparison(
    pre_df=pre_weekly_count_brooklyn,
    post_df=post_weekly_count_brooklyn,
    complaint="Loud Music/Party"
)

################################################################################
##### Plot 6: Spatial Comparison of Loud Music and Airbnb Change (Fishnet) #####
################################################################################
############################## Manhattan Study Area ############################
# Load cleaned datasets
final_merged_df = pd.read_csv("../data/processed/Cleaned_data_11211.csv")
Manhattan_final_merged_df = pd.read_csv("../data/processed/Cleaned_data_10002.csv")

# Ensure spatial alignment (all to EPSG:2263)
LMP_pre_policy_manhattan = pre_policy_manhattan[pre_policy_manhattan['Descriptor'] == 'Loud Music/Party'].to_crs(epsg=2263)
LMP_post_policy_manhattan = post_policy_manhattan[post_policy_manhattan['Descriptor'] == 'Loud Music/Party'].to_crs(epsg=2263)
zipcode_boundary_manhattan = zipcode_boundary_10002.to_crs(epsg=2263)
changed_airbnb_manhattan = gdf_10002[gdf_10002['rental_change_type'] == 'Longer Rental'].to_crs(epsg=2263)

# Run fishnet spatial change visualization
plot_spatial_change_comparison(
    pre_gdf=LMP_pre_policy_manhattan,
    post_gdf=LMP_post_policy_manhattan,
    airbnb_gdf=changed_airbnb_manhattan,
    boundary_gdf=zipcode_boundary_manhattan,
    title="Grid-based Pre-post Count Difference in Manhattan Study Area"
)

############################### Brooklyn Study Area ############################
# Ensure spatial alignment (all to EPSG:2263)
LMP_pre_policy_brooklyn = pre_policy_brooklyn[pre_policy_brooklyn['Descriptor'] == 'Loud Music/Party'].to_crs(epsg=2263)
LMP_post_policy_brooklyn = post_policy_brooklyn[post_policy_brooklyn['Descriptor'] == 'Loud Music/Party'].to_crs(epsg=2263)
zipcode_boundary_brooklyn = zipcode_boundary.to_crs(epsg=2263)
changed_airbnb_brooklyn = gdf[gdf['rental_change_type'] == 'Longer Rental'].to_crs(epsg=2263)

# Run fishnet spatial change visualization
plot_spatial_change_comparison(
    pre_gdf=LMP_pre_policy_manhattan,
    post_gdf=LMP_post_policy_manhattan,
    airbnb_gdf=changed_airbnb_brooklyn,
    boundary_gdf=zipcode_boundary_brooklyn,
    title="Grid-based Pre-post Count Difference in Brooklyn Study Area"
)
