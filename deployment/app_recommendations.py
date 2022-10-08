import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time
from numpy.linalg import norm

# formatting, create 3 columns
col1, col2, col3 = st.columns([1,0.25,1])

# Title
col1.markdown(" # Airbnb Recommendations")

# Subtitle
col1.markdown(" Select an Airbnb listing and get a recommendation for similar listings.")

## UPLOAD THE DATAFRAMES ##
# ----------------------------------------------- #

# load in sd_trans dataframe to be transformed
sd_trans = pd.read_csv('sd_trans', index_col = 0)

# load in url_listings dataframe to be joined
sd_listings_url = pd.read_csv('url_listings', index_col = 0)

## UPLOAD THE SELECTION DFS ## 
# ----------------------------------------------- #

# load in sd_pp, FOR SELECTION OF URL WITHIN THE PREPROCESSED DF
sd_pp = pd.read_csv('sd_pp', index_col = 0)
# load in sd_clustered
sd_clustered = pd.read_csv('sd_clustered', index_col = 0)

# merge url listings with sd_trans
sd_merged = sd_listings_url.join(sd_trans)

## GET URL LISTING FROM PP DATASET ##
# ----------------------------------------------- #
# SELECT THE LISTING FROM UNPROCESSED DATASET 
# get sd_clustered and merge with urls on index
sd_clustered = sd_clustered.join(sd_listings_url)

# select a listing from sd_merged
selected_listing = st.selectbox("Choose a listing below:", sd_merged.listing_url)

# based on selected listing, get the index from sd_pp
index_value = sd_merged.listing_url[sd_merged.listing_url == str(selected_listing)].index[0]
selected_listing_df = pd.DataFrame(sd_pp.iloc[index_value]).T

## TRANSFORMS THE DATASET INTO A PREPROCESSED SET ## 
# ----------------------------------------------- #

# unpickle and load in column transformer
ct = joblib.load("column_transformer.pkl")

## GET RECOMMENDATION BASED ON SELECTED LISTING ##
# ----------------------------------------------- #

# get a recommendation based on url selection
def get_recommendations(df, listing):
    """
    Takes in preprocessed dataframe and selected listing as inputs and gives top 5
    recommendations based on cosine similarity. 
    """
    # reset the index
    df = df.reset_index(drop = 'index')
    
    # convert single listing to an array
    listing_array = listing.values

    # convert all listings to an array
    df_array = df.values
    
    # get arrays into a single dimension
    A = np.squeeze(np.asarray(df_array))
    B = np.squeeze(np.asarray(listing_array))
    
    # compute cosine similarity 
    cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
    # add similarity into recommendations df and reset the index
    rec = sd_clustered.copy().reset_index(drop = 'index')
    rec['similarity'] = pd.DataFrame(cosine).values
    
    # reorder column names
    rec = rec[['listing_url', 'similarity', 'cluster_label', 'latitude', 'longitude',
       'neighbourhood_cleansed', 'zipcode', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
       'nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee',
       'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
       'host_response_time', 'host_response_rate', 'host_is_superhost',
       'host_total_listings_count', 'host_has_profile_pic',
       'host_identity_verified', 'number_of_reviews', 'number_of_stays',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'requires_license', 'instant_bookable',
       'is_business_travel_ready', 'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification']]

    # drop extra columns
    
    # rename columns
    rec = rec.rename(columns = {"neighbourhood_cleansed": "Neighborhood"})

    # selected listing
    selection = st.dataframe(rec.sort_values(by = ['similarity'], ascending = False)[0:1])
    
    if selection: 
        st.write("We recommend booking at these similar stays:")

    # sort by top 5 descending
    recommended_listings = st.dataframe(rec.sort_values(by = ['similarity'], ascending = False)[1:6])

    return selection, recommended_listings

# get recommendation
get_recommendations(sd_pp, selected_listing_df)

## GET A DATAFRAME BASED ON USER SELECTED PARAMETERS ## 
# ----------------------------------------------- #

option = st.selectbox(
    'Neighborhood',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)



