import numpy as np
import pickle
import pandas as pd
import streamlit as st
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

# # load in sd_trans dataframe to be transformed
# sd_trans = pd.read_csv('sd_trans', index_col = 0)

# # load in url_listings dataframe to be joined
# sd_listings_url = pd.read_csv('url_listings', index_col = 0)

# ## UPLOAD THE SELECTION DFS ## 
# # ----------------------------------------------- #

# # load in sd_pp, FOR SELECTION OF URL WITHIN THE PREPROCESSED DF
# sd_pp = pd.read_csv('sd_pp', index_col = 0)

# # load in sd_clustered
# sd_clustered = pd.read_csv('sd_clustered', index_col = 0)

# # merge url listings with sd_trans
# sd_merged = sd_listings_url.join(sd_trans)

# ## GET URL LISTING FROM PP DATASET ##
# # ----------------------------------------------- #
# # SELECT THE LISTING FROM UNPROCESSED DATASET 
# # get sd_clustered and merge with urls on index
# sd_clustered = sd_clustered.join(sd_listings_url)

# # select a listing from sd_merged
# selected_listing = st.selectbox("Choose a listing below:", sd_merged.listing_url)

# # based on selected listing, get the index from sd_pp
# index_value = sd_merged.listing_url[sd_merged.listing_url == str(selected_listing)].index[0]
# selected_listing_df = pd.DataFrame(sd_pp.iloc[index_value]).T

# ## TRANSFORMS THE DATASET INTO A PREPROCESSED SET ## 
# # ----------------------------------------------- #
# # # unpickle and load in column transformer
# # ct = joblib.load("column_transformer.pkl")

# with open('column_transformer.pkl', 'rb') as f:
#     ct = pickle.load(f)

# ## GET RECOMMENDATION BASED ON SELECTED LISTING ##
# # ----------------------------------------------- #

# # get a recommendation based on url selection
# def get_recommendations(df, listing):
#     """
#     Takes in preprocessed dataframe and selected listing as inputs and gives top 5
#     recommendations based on cosine similarity. 
#     """
#     # reset the index
#     df = df.reset_index(drop = 'index')
    
#     # convert single listing to an array
#     listing_array = listing.values

#     # convert all listings to an array
#     df_array = df.values
    
#     # get arrays into a single dimension
#     A = np.squeeze(np.asarray(df_array))
#     B = np.squeeze(np.asarray(listing_array))
    
#     # compute cosine similarity 
#     cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
#     # add similarity into recommendations df and reset the index
#     rec = sd_clustered.copy().reset_index(drop = 'index')
#     rec['similarity'] = pd.DataFrame(cosine).values

#     # simplify the dataframe and keep only necessary columns
#     rec = rec[['listing_url', 'similarity', 'cluster_label', 'neighbourhood_cleansed',
#                 'property_type', 'room_type', 'accommodates', 'bathrooms', 'beds',
#                 'nightly_price', 'review_scores_rating']]
    
#     # rename columns
#     rec = rec.rename(columns = {'listing_url': 'URL', 
#                                 'similarity': 'Similarity Score',
#                                 'cluster_label': 'Listing Category',
#                                 'neighbourhood_cleansed': 'Neighborhood',
#                                 'property_type': 'Property Type',
#                                 'room_type': 'Room Type',
#                                 'accommodates': 'Accommodates',
#                                 'bathrooms': ' Bathrooms',
#                                 'beds': 'Beds',
#                                 'nightly_price': 'Nightly Price',
#                                 'review_scores_rating': 'Review Rating'})

#     # rename the cluster_label AKA Listing Category column values
#     rec = rec.replace({'Listing Category': {0:'Popular High End',
#                                             1:'Highly Rated & Moderately Priced',
#                                             2:'Diverse & Moderately Priced',
#                                             3:'Favorable & Budget Friendly',
#                                             4:'Unfavorable & Poorly Rated'}})
    

#     # selected listing
#     selection = st.dataframe(rec.sort_values(by = ['Similarity Score'], ascending = False)[0:1])
    
#     if selection: 
#         st.write("We recommend booking at these similar stays:")

#     # sort by top 5 descending
#     recommended_listings = st.dataframe(rec.sort_values(by = ['Similarity Score'], ascending = False)[1:6])

#     return selection, recommended_listings

# # get recommendation
# get_recommendations(sd_pp, selected_listing_df)

# ## GET A DATAFRAME BASED ON USER SELECTED PARAMETERS FOR A SIMPLIFIED RECOMMENDATION ## 
# # ----------------------------------------------- #

# # load in the simplified dataset
# sd_simplified = pd.read_csv('sd_simplified', index_col = 0)


# ## GET USER INPUTS ## 
# # ----------------------------------------------- #
# neighborhood = st.selectbox('Neighborhood:',(sd_simplified['neighbourhood_cleansed'].unique()))
# property = st.selectbox('Property Type:',(sd_simplified['property_type'].unique()))
# room = st.selectbox('Room Type:',(sd_simplified['room_type'].unique()))
# accommodation = st.selectbox('Accommodation:',(sd_simplified['accommodates'].unique()))
# bathrooms = st.selectbox('Bathrooms:',(sd_simplified['bathrooms'].unique()))
# beds = st.selectbox('Number of Beds:',(sd_simplified['beds'].unique()))
# price = st.slider('Minimum Nightly Price ($):', int(sd_simplified['nightly_price'].min().item()),  # min
#                                             int(sd_simplified['nightly_price'].quantile(0.75).item()),  # max
#                                             int(sd_simplified['nightly_price'].median().item()),
#                                             step = 1) # start point

# rating = st.slider('Minimum Rating:',  int(sd_simplified['review_scores_rating'].min().item()),  # min
#                                             int(sd_simplified['review_scores_rating'].quantile(0.75).item()),  # max
#                                             90, # start at 90
#                                             step = 1) # start point

# # store inputs into df
# column_names = ['Neighborhood', 'Property Type', 'Room Type', 'Accommodation', 'Bathrooms', 
#                 'Number of Beds', 'Minimum Nightly Price ($)', 'Minimum Rating']
# user_inputs = pd.DataFrame([neighborhood, property, room, accommodation, bathrooms, beds, price], 
#                             columns = column_names)

# # transform the simplified dataset
# sd_simplified_pp = pd.Dataframe(ct.fit_transform(sd_simplified))


# ## GET RECOMMENDATION BASED ON USER INPUTS ##
# # ----------------------------------------------- #

# def get_simplified_recommendations(df, user_inputs):
#     """
#     Takes in preprocessed dataframe and preprocessed user inputs df and gives top 5
#     recommendations based on cosine similarity. 
#     """
#     # reset the index
#     df = df.reset_index(drop = 'index')
    
#     # transform the user_inputs dataframe into preprocessed dataset
#     user_inputs_df_pp = pd.DataFrame(ct.transform(user_inputs))
    
#     # convert single listing to an array
#     listing_array = user_inputs_df_pp.values
#     # convert all listings to an array
#     df_array = df.values
    
#     # get arrays into a single dimension
#     A = np.squeeze(np.asarray(df_array))
#     B = np.squeeze(np.asarray(listing_array))
    
#     # compute cosine similarity 
#     cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
#     # add similarity into recommendations df and reset the index
#     rec = sd_simplified.copy().reset_index(drop = 'index')
#     rec['similarity'] = pd.DataFrame(cosine).values
    
#     # add in listings_urls
#     # merge on index
#     rec = rec.join(sd_listings_url)
    
#     # reorder column names
#     rec = rec[['listing_url', 'similarity', 'neighbourhood_cleansed', 'property_type', 
#                'room_type', 'accommodates', 'bathrooms', 'beds', 'nightly_price', 'review_scores_rating']]
    
#     # sort by top 5 descending
#     return rec.sort_values(by = ['similarity'], ascending = False).head(5)

# # get recommendation
# get_simplified_recommendations(sd_simplified_pp, user_inputs)


