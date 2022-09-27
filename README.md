# Airbnb Recommendations & Natural Language Processing (NLP) Analysis

Airbnb is an online marketplace that is focussed on connecting people who rent out their homes with people who are looking for accommodations around the world. If you're like me and love to travel, Airbnb provides a cheaper alternative to hotels while also offering an eccentric experience that adds value to vacations. 

![Screen Shot 2022-09-27 at 11 33 26 AM](https://user-images.githubusercontent.com/86889081/192570464-69968dc0-4e18-45ce-b7fe-92ad059d9acb.png)

I personally love traveling to the west coast and finding unique Airbnb listings. Sometimes I'd find myself wanting to go back but have trouble looking for a similar experience. There is currently no system in place where Airbnb will provide (or recommend) similar homes I have previously stayed in. 

So that got me thinking; **Can I develop a machine learning recommendation system that can provide recommendations for similar listings that I have stayed in? Additionally, what can we learn from text descriptions for Airbnb listings?**

For this analysis, the following machine learning concepts were practiced:

### Recommendation System:
A recommendation system helps users find compelling content in a large mass of data. For example, a machine learning recommendation model determines how similar listings are to other listings then provides recommendations based on similarity. A recommendation engine can display items that users might not have been to search for on their own. This is the basis of most recommendation systems and the inspiration for this project. 

### Natural Language Processing (NLP): 
NLP makes it possible for computers to understand the human language. NLP analyzes the grammatical structure of sentences and uses algorithms to find and extract the individual meaning of words. We are most accustomed to NLP schemes with virtual assistants like Alexa or Siri, that translate spoken text into meaning that the machines can understand. For this project, we will be performing some NLP analysis on Airbnb listing descriptions and incoorporating a sentiment analysis in the recommendation engine. 

## Data Understanding
San Diego is a city that I frequently visit on the west coast. Therfore, I was compelled to work on a dataset of Airbnb listings within the area. 

The dataset for this project consists of over 13,000 rows of data for <a href = 'https://data.world/ajsanne/san-diego-airbnb'> San Diego Airbnb Listings as of August 2019 </a> and publicly sourced from data.world via Inside Airbnb.

There are 75 features and in general, consist of the following: 
- unique listing ids & urls,
- text descriptions of the listing (name, summary, space, neighborhood overview, amenities, house rules, city, neighborhood, property type, room_type, bed type, etc),
- text descriptions of the host (host name, about, response time, location, etc)
- numerical descriptions (host response rate/time, number of bathrooms, bedrooms, accommodation, price, number of stays, number of reviews, review scores, etc)
- binary values (instant bookable, license requirements, host identity verfication status, etc)



