import pandas as pd
import geopandas as gpd

import os
from shapely.geometry import Point
from shapely.wkt import loads


# Process Airbnb data
def process_airbnb_data():
    # Select the features would use.
    cols = ['id', 'name', 'neighbourhood_group', 'latitude','longitude', 'room_type', 'price', 'minimum_nights','neighbourhood']

    # Read data. Process each file separately
    df_2023 = pd.read_csv('data/raw/NYC-Airbnb-2023.csv',usecols=cols,low_memory=False)
    df_2024 = pd.read_csv('data/raw/NYC-Airbnb-2024.csv',usecols=cols,low_memory=False)
    nyc_zipcode_boundaries = pd.read_csv("data/raw/nyc_zipcode_boundaries.csv")

    # Area:Filter only Brooklyn listings
    df_2023_filtered = df_2023[df_2023['neighbourhood_group'] == 'Brooklyn']
    df_2024_filtered = df_2024[df_2024['neighbourhood_group'] == 'Brooklyn']

    # Section 1: Data Preprocessing and Merging
    df_2024_selected = df_2024_filtered[['id', 'minimum_nights', 'price']]
    merged_df = pd.merge(df_2023_filtered, df_2024_selected, on='id', how='inner')

    # Section 2: Geospatial Data Preparation (Convert to GeoDataFrame)
    merged_df_1 = merged_df[['id', 'latitude', 'longitude']].copy()
    merged_df_1['geometry'] = merged_df_1.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    geo_merged_df_1 = gpd.GeoDataFrame(merged_df_1, geometry='geometry')

    # Section 3: Extract Target Region (ZIP Code 11211)
    zip_code_target = nyc_zipcode_boundaries[nyc_zipcode_boundaries["MODZCTA"] == 11211]
    zip_code_target = zip_code_target.copy()
    if 'the_geom' in zip_code_target.columns:
        zip_code_target['geometry'] = zip_code_target['the_geom'].apply(lambda geom: loads(geom) if isinstance(geom, str) else geom)
    zip_code_target = gpd.GeoDataFrame(zip_code_target, geometry='geometry')
    zip_code_target = zip_code_target.set_crs("EPSG:4326")
    geo_merged_df_1 = geo_merged_df_1.set_crs("EPSG:4326")
    geo_merged_df_1 = geo_merged_df_1.to_crs(zip_code_target.crs)

    # Section 4: Spatial Clipping and Final Output
    clipped_df = gpd.clip(geo_merged_df_1, zip_code_target)
    final_merged_df = pd.merge(merged_df, clipped_df, on='id', how='inner')
    final_merged_df = final_merged_df.drop(columns=['longitude_y', 'latitude_y'])
    os.makedirs("../data/processed", exist_ok=True)
    final_merged_df.to_csv("data/processed/Cleaned_data_11211.csv", index=False)

    # Manhattan Region(10002) Processing Data
    Manhattan_2023 = df_2023[df_2023['neighbourhood_group'] == 'Manhattan']
    Manhattan_2024 = df_2024[df_2024['neighbourhood_group'] == 'Manhattan']
    Manhattan_2023_cleaned = Manhattan_2023.drop_duplicates(subset='id')
    Manhattan_2024_cleaned = Manhattan_2024.drop_duplicates(subset='id')
    Manhattan_2024_selected = Manhattan_2024_cleaned[['id', 'minimum_nights', 'price']]
    Manhattan_merged_df = pd.merge(Manhattan_2023_cleaned, Manhattan_2024_selected, on='id', how='inner')

    Manhattan_merged_df_1 = Manhattan_merged_df[['id', 'latitude', 'longitude']].copy()
    Manhattan_merged_df_1['geometry'] = Manhattan_merged_df_1.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    geo_Manhattan_merged_df_1 = gpd.GeoDataFrame(Manhattan_merged_df_1, geometry='geometry')

    Manhattan_zip_code_target = nyc_zipcode_boundaries[nyc_zipcode_boundaries["MODZCTA"] == 10002]
    Manhattan_zip_code_target = Manhattan_zip_code_target.copy()
    if 'the_geom' in Manhattan_zip_code_target.columns:
        Manhattan_zip_code_target['geometry'] = Manhattan_zip_code_target['the_geom'].apply(
            lambda geom: loads(geom) if isinstance(geom, str) else geom
        )
    Manhattan_zip_code_target = gpd.GeoDataFrame(Manhattan_zip_code_target, geometry='geometry')
    Manhattan_zip_code_target = Manhattan_zip_code_target.set_crs("EPSG:4326")
    geo_Manhattan_merged_df_1 = geo_Manhattan_merged_df_1.set_crs("EPSG:4326")
    geo_Manhattan_merged_df_1 = geo_Manhattan_merged_df_1.to_crs(Manhattan_zip_code_target.crs)

    Manhattan_clipped_df = gpd.clip(geo_Manhattan_merged_df_1, Manhattan_zip_code_target)
    Manhattan_final_merged_df = pd.merge(Manhattan_merged_df, Manhattan_clipped_df, on='id', how='inner')
    Manhattan_final_merged_df = Manhattan_final_merged_df.drop(columns=['longitude_y', 'latitude_y'])
    Manhattan_final_merged_df.to_csv("data/processed/Cleaned_data_10002.csv", index=False)
    return merged_df, final_merged_df, Manhattan_merged_df, Manhattan_final_merged_df



# Process noise data
def process_noise_data():
    file_path_brooklyn = 'data/raw/Noise_Complaints_williamsburg_22-24.csv'
    file_path_manhattan = 'data/raw/Noise_Complaints_chinatown_22-24.csv'

    # Load Brooklyn noise data
    noise_brooklyn = pd.read_csv(file_path_brooklyn)

    # Convert ZIP code to integer and filter for study area ZIPs
    noise_brooklyn['Incident Zip'] = noise_brooklyn['Incident Zip'].astype('Int64')
    studyarea_noise_brooklyn = noise_brooklyn[noise_brooklyn['Incident Zip'].isin([11211, 11249])]

    # Keep only residential noise complaints
    noise_res_brooklyn = studyarea_noise_brooklyn[studyarea_noise_brooklyn['Complaint Type'] == 'Noise - Residential']

    # Select relevant columns
    columns = ['Created Date', 'Closed Date', 'Complaint Type', 'Descriptor', 'Incident Zip',
               'Borough', 'Location Type', 'Latitude', 'Longitude']
    filtered_noise_brooklyn = noise_res_brooklyn[columns]

    # Drop rows with missing location and remove duplicates
    noise_brooklyn_cleaned = filtered_noise_brooklyn.dropna(subset=['Incident Zip', 'Latitude', 'Longitude']).drop_duplicates()

    # Create GeoDataFrame with point geometries (EPSG:4326 = WGS84)
    gdf_brooklyn = gpd.GeoDataFrame(
        noise_brooklyn_cleaned,
        geometry=gpd.points_from_xy(noise_brooklyn_cleaned['Longitude'], noise_brooklyn_cleaned['Latitude']),
        crs='EPSG:4326'
    )

    # Parse timestamp and convert CRS to EPSG:2263 (NY State Plane)
    gdf_brooklyn['datetime_fmt'] = pd.to_datetime(gdf_brooklyn['Created Date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    gdf_brooklyn = gdf_brooklyn.to_crs(epsg=2263)
    gdf_brooklyn = gdf_brooklyn.sort_values('datetime_fmt')

    # Extract date and time
    gdf_brooklyn['date_fmt'] = gdf_brooklyn['datetime_fmt'].dt.date
    gdf_brooklyn['time_fmt'] = gdf_brooklyn['datetime_fmt'].dt.time

    # Extract month and hour
    gdf_brooklyn['month_fmt'] = gdf_brooklyn['datetime_fmt'].dt.to_period('M')
    gdf_brooklyn['hour'] = gdf_brooklyn['datetime_fmt'].dt.hour


    # Filter to target date range
    start = pd.to_datetime('2022-09-01')
    end = pd.to_datetime('2024-08-31')
    cutoff = pd.to_datetime("2023-09-01")
    gdf_brooklyn = gdf_brooklyn[(gdf_brooklyn['datetime_fmt'] >= start) & (gdf_brooklyn['datetime_fmt'] <= end)]

    # Split into pre-policy and post-policy periods
    pre_policy_brooklyn = gdf_brooklyn[gdf_brooklyn['datetime_fmt'] < cutoff]
    post_policy_brooklyn = gdf_brooklyn[gdf_brooklyn['datetime_fmt'] >= cutoff]

    # Load Manhattan noise data
    noise_manhattan = pd.read_csv(file_path_manhattan)

    # Filter only residential complaints
    noise_res_manhattan = noise_manhattan[noise_manhattan['Complaint Type'] == 'Noise - Residential']
    filtered_noise_manhattan = noise_res_manhattan[columns]
    noise_manhattan_cleaned = filtered_noise_manhattan.dropna(subset=['Incident Zip', 'Latitude', 'Longitude']).drop_duplicates()

    # Create GeoDataFrame
    gdf_manhattan = gpd.GeoDataFrame(
        noise_manhattan_cleaned,
        geometry=gpd.points_from_xy(noise_manhattan_cleaned['Longitude'], noise_manhattan_cleaned['Latitude']),
        crs='EPSG:4326'
    )

    # Parse timestamp and convert CRS
    gdf_manhattan['datetime_fmt'] = pd.to_datetime(gdf_manhattan['Created Date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    gdf_manhattan = gdf_manhattan.to_crs(epsg=2263)
    gdf_manhattan = gdf_manhattan.sort_values('datetime_fmt')

    # Extract date and time
    gdf_manhattan['date_fmt'] = gdf_manhattan['datetime_fmt'].dt.date
    gdf_manhattan['time_fmt'] = gdf_manhattan['datetime_fmt'].dt.time

    # Extract month and hour
    gdf_manhattan['month_fmt'] = gdf_manhattan['datetime_fmt'].dt.to_period('M')
    gdf_manhattan['hour'] = gdf_manhattan['datetime_fmt'].dt.hour


    # Filter to target date range
    gdf_manhattan = gdf_manhattan[(gdf_manhattan['datetime_fmt'] >= start) & (gdf_manhattan['datetime_fmt'] <= end)]

    # Split into pre-policy and post-policy periods
    pre_policy_manhattan = gdf_manhattan[gdf_manhattan['datetime_fmt'] < cutoff]
    post_policy_manhattan = gdf_manhattan[gdf_manhattan['datetime_fmt'] >= cutoff]

    # Save processed datasets to disk
    os.makedirs("data/processed", exist_ok=True)
    gdf_brooklyn.to_csv("data/processed/cleaned_data_brooklyn_all.csv", index=False)
    gdf_manhattan.to_csv("data/processed/cleaned_data_manhattan_all.csv", index=False)
    pre_policy_brooklyn.to_csv("data/processed/cleaned_data_brooklyn_pre.csv", index=False)
    post_policy_brooklyn.to_csv("data/processed/cleaned_data_brooklyn_post.csv", index=False)
    pre_policy_manhattan.to_csv("data/processed/cleaned_data_manhattan_pre.csv", index=False)
    post_policy_manhattan.to_csv("data/processed/cleaned_data_manhattan_post.csv", index=False)

    return gdf_brooklyn, gdf_manhattan, pre_policy_brooklyn, post_policy_brooklyn, pre_policy_manhattan, post_policy_manhattan
