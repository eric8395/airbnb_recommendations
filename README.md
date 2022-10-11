# Airbnb Recommendations, Clustering, & Natural Language Processing (NLP) Analysis

Airbnb is an online marketplace that is focussed on connecting people who rent out their homes with people who are looking for accommodations around the world. If you're like me and love to travel, Airbnb provides a cheaper alternative to hotels while also offering an eccentric experience that adds value to vacations. 

![Screen Shot 2022-09-27 at 11 33 26 AM](https://user-images.githubusercontent.com/86889081/192570464-69968dc0-4e18-45ce-b7fe-92ad059d9acb.png)

I personally love traveling to the west coast and finding unique Airbnb listings. Sometimes I'd find myself wanting to go back but have trouble looking for a similar experience. There is currently no system in place where Airbnb will provide (or recommend) similar homes I have previously stayed in. 

So that got me thinking:
- **Can I develop a machine learning recommendation system that can provide recommendations for similar listings that I have stayed in? Additionally, what can we learn from text descriptions for Airbnb listings?**

## Web Application Deployment 
I developed a web application deployed through Streamlit that provides a user friendly interface for generating Airbnb recommendations. Users have the ability to either input a recent stay or find their own personalized stay with custom inputs. Recommendations are subsequently generated for similar Airbnb listings. Feel free to check it out for yourself below!

**Application:** <a href = 'https://eric8395-airbnb-recommendations-app-app-fb49vg.streamlitapp.com/'>Discover San Diego!</a>

<a href = 'https://github.com/eric8395/airbnb_recommendations_app'>View Source Code</a>

<img width="913" alt="Screen Shot 2022-10-10 at 3 38 16 PM" src="https://user-images.githubusercontent.com/86889081/194940447-f20d7f8b-75e4-462a-8cf0-34a849666be5.png">

## Machine Learning Concepts
For this analysis, the following machine learning concepts were practiced:

### Recommendation System:
A recommendation system helps users find compelling content in a large mass of data. For example, a machine learning recommendation model determines how similar listings are to other listings then provides recommendations based on similarity. A recommendation engine can display items that users might not have been to search for on their own. This is the basis of most recommendation systems and the inspiration for this project. 

### Cluster Analysis: 
Clustering is a type of unsupervised learning method, whereby information among groups are gained from unlabled data. Clustering allows machines to pick up on similar or dissimilar patterns that would otherwise be difficult to discern. We can use clustering as a means to group Airbnb listings when it comes to finding recommendations of similar listings. 

### Natural Language Processing (NLP): 
NLP makes it possible for computers to understand the human language. NLP analyzes the grammatical structure of sentences and uses algorithms to find and extract the individual meaning of words. We are most accustomed to NLP schemes with virtual assistants like Alexa or Siri, that translate spoken text into meaning that the machines can understand. For this project, we will be performing some NLP analysis on Airbnb listing descriptions and incoorporating a sentiment analysis in the recommendation engine. 

## Data Understanding
San Diego is a city that I frequently visit on the west coast. Therfore, I was compelled to work on a dataset of Airbnb listings within the area. 

The dataset for this project consists of over 13,000 rows of data for <a href = 'https://data.world/ajsanne/san-diego-airbnb'> San Diego Airbnb Listings as of August 2019 </a> and publicly sourced from data.world via Inside Airbnb.

There are 75 features and in general, consist of the following: 
- unique listing ids & urls,
- text descriptions of the listing (name, summary, space, neighborhood overview, amenities, house rules, city, neighborhood, property type, room type, bed type, etc),
- text descriptions of the host (host name, about, response time, location, etc)
- numerical descriptions (host response rate/time, number of bathrooms, bedrooms, accommodation, price, number of stays, number of reviews, review scores, etc)
- binary values (instant bookable, license requirements, host identity verfication status, etc)

### Data Cleaning & Preprocessing
In general, the following steps below describe the major data cleaning and preprocessing performed before conducting analysis. For more detailed steps, please refer to the *'Data Cleaning'* and *'Preprocessing, Data Visualization, Clustering'* notebooks found in this repository. 

- **Handling Missing Values:** There were many missing values discovered in dataset. For example, `host_response_time` contained over 2,100 rows of missing entries. Since this was a categorical ordinal column, these missing values were imputed with an 'N/A' value to represent Airbnb hosts who have not responded back to hostees. Other numerical missing values such as `security_deposits` were imputed with the value 0 (assuming that a security deposit was not needed for the listings). 

- **Encoding Categorical Features and Values:** Categorical features were split into ordinal and nominal features. Ordinal features (columns where the values have a structured order) consisted of `host_response_time` and `cancellation_policy` and were encoded using an `OrdinalEncoder`. Nominal features (columns where values have no order of precedence) consisted of all other categorical features (ie. `property_type`) and were one hot encoded. 

- **Standard Scaling:** All other numerical columns consisting of integer and float values were subsequently scaled using a `StandardScaler`. 

In total, the preprocessed dataset consisted of **13,039 listings and 240 features**. 

## Clustering Analysis
<a href = 'https://umap-learn.readthedocs.io/en/latest/' /> A Uniform Manifold Approximation and Project (UMAP) </a> dimensionality reduction technique was leveraged to create a clustering visualization of all data points following preprocessing. 

Clustering labels were constructed using a MiniBatch KMeans iterating through the preprocessed dataset to determine optimum cluster size. A total of 5 unique cluster groups were generated with labels assigned to each individual listing. 

Below is a snippet image of **San Diego Airbnb Listings Embedding via UMAP** whereby users can hover over each individual data point to obtain a better understanding of the features associated within each cluster. Visualization is generated using Bokeh. 

<p align="center">
<img width="495" alt="bokeh clusters" src="https://user-images.githubusercontent.com/86889081/192680072-ec7387ea-086e-4842-916a-7d11646cbfc8.png">
<p align="center"> 

The following observations about each cluster group can be generally summarized as follows:
  
**Cluster Label 0 (Red) - Favorable high end listings**
- Favorable and wide range of review rating. Most expensive listings and mostly consist of entire home room types.
  
**Cluster Label 1 (Orange) - Favorable highly rated & moderately priced listings**
- Popular group, generally > 90 review ratings, relatively inexpensive. Mostly houses or private rooms, wide range of property types.
  
**Cluster Label 2 (Yellow) - Favorable moderately priced diverse listings**
- Most popular group, mostly favorable ratings. Relatively low priced. Wide range of property types.
  
**Cluster Label 3 (Green) - Favorable and least expensive listings**
- Popular group and wide range of review rating. Least expensive group. Wide range of property types.
  
**Cluster Label 4 (Purple) - Unfavorable listings**
- Least popular group and lowest rated listings.

## Natural Language Processing - Visualizing Text Descriptions

Wordcloud visualizations were constructed for each text column in the Airbnb Listings dataset. The text columns were preprocessed and normalized as follows:
- Missing values are imputed with 'blank' text
- Text in each column are tokenized. Tokenization is a process by which the text is broken down into smaller units and subwords. Stopwords and other text without much value were also removed during this step. 
  
<p align="center">
<img width="400" src="https://github.com/eric8395/airbnb_recommendations/blob/main/images/wordcloud_summary.png">

**Findings**
- Most listings tend to be described as scenic and picturesque by the beach (or some variation of paradise).
- A lot of listings are ironically "hidden".
- Must have wifi, tv, parking, and large beds!
- The hosts must have a lot of spare time to rent out Airbnbs as their side jobs are also involved in entertainment.

### Average Sentiment by Feature  

A sentiment analysis was also performed and to gather further insights about how Airbnb listings are generally described. The process of understanding sentiment scores is described as follows:

- **TextBlob Module:** Allows for the ability to place a score on sentiment of words based on where it is in a sentence.
- **Sentiment Labels:** Each word in a corpus is labeled in terms of polarity and subjectivity.
- **Polarity:** How positive or negative a word is; -1 is most negative, +1 is most positive
- **Subjectivity:** How subjective, or opinionated a word is; 0 is fact, +1 is an opinion

<p align="center">
<img width="664" alt="sentiment chart" src="https://user-images.githubusercontent.com/86889081/193640408-fa1c0057-8dd2-4351-a7ff-cde5e42d5330.png">

**Takeaways**
- Airbnb listings tend to be positive when it comes to descriptions and summaries. This makes sense, hosts want to encourage people to stay at their Airbnb and having a positive description is beneficial. However, these descriptions tend to be grounded in opinion.
- Factually based columns such as access, notes, and transit are unsurprisingly factual.
- Interesting to note that amenities are considered very opinionated. One would expect that amenities would be more grounded in facts.

## Next Steps & Recommendation System Limitations
Having constructed a recommendation system based on user based content filtering, there are a few nuances as part of this project:
- URL links are not entirely up to date. Since the data for this recommendation system consists of 2019 data, there are instances where the url links to the actual Airbnb listing do not exist anymore. However, conceptually, the recommendation system is still effective when it comes to analysis of other content based features. 

- While the dataset contained textual data, the sentiment analysis was limited to description of listings by the hosts. Moving forward, I'd like to incoorporate a sentiment analysis of users who have previously stayed at a listing (ie. user reviews). With this analysis, there is potential to create a recommendation system of ideal Airbnb listings based on the sentiment analysis of user reviews. 

