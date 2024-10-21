# import python packages
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordLemmatizer = WordNetLemmatizer()

# Get the finalized artifacts i.e. TFIDF vectorizer, the sentiment prediction model
# and the recommendation ratings
tfidf_transformer = pickle.load(open('pickles/tfidf_vectorizer.pkl','rb'))
sentiment_model = pickle.load(open('pickles/lr.pkl', 'rb'))
user_ratings = pickle.load(open('pickles/user_final_rating.pkl', 'rb'))

# Get the review data for all users and products
df_reviews = pd.read_csv('sample30.csv')

# Impute missing manufacturer 
df_reviews.loc[df_reviews[pd.isnull(df_reviews.manufacturer)].index,'manufacturer'] = 'Summit Entertainment'

# Get rated and valid user names
rated_users = user_ratings[(user_ratings != 0).any(axis=1)].index.to_list()
all_users = user_ratings.index.to_list()


def get_lemmatized(in_text):
    ''' This function handles the lemmatized of the given text
      - Remove any unwanted characters including punctuation
      - Convert the text into lower case
      - Tokenize the text into words
      - Get the lemmas for the tokens after ignoring stopwords
      - Finally get the result text by joining the lemmas
    '''
    text = re.sub(r"[^a-zA-Z0-9\s,']",'',in_text)
    text = text.lower()
    text = text.split()
    text = [wordLemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return " ".join(text)


def valid_user(user):
    ''' This function validates the provided user and gives feedback
      - if user is invalid
      - if user is valid yet has no recommendations yet
      - if user has recommendations
    '''
    if user not in all_users:
        print("User does not exist. Please enter a valid user.")
        return 0
    elif user not in rated_users:
        print("No recommendations yet for this user. Choose a different user")
        return 1
    else:
        return 2
  

def get_recommended_products(user):
    ''' This function returns top 5 recommended products for the given user
      - checks if user is valid
      - if valid, get the top 20 recommended product based on user ratings
      - then, fine tunes the result by shortlisting top 5 based on product sentiment (positive)
    '''
    # validate user
    ok_code = valid_user(user)

    if ok_code == 2:
        print("Selected user: " + user)

        # get top 20 recommended products for the user
        top20_products = pd.DataFrame(user_ratings.loc[user].sort_values(ascending=False)[0:20])

        # Get all reviews for the top 20 products recommended
        df_reviews_20 = df_reviews.loc[(df_reviews.name.isin(top20_products.index.tolist())),['name','brand','manufacturer','reviews_text']]

        # Filter top 5 of the 20 recommended using sentiment prediction
        df_top5 = get_top5_recommended_products(df_reviews_20)
        
        return df_top5.loc[:,['name','brand','manufacturer']]
    
    elif ok_code == 0:
       return "User does not exist. Please enter a valid user."
    else:
       return "No recommendations yet for this user. Choose a different user."


def get_top5_recommended_products(df):
    ''' This function returns top 5 recommended products from the given products
      - it uses the sentiment model to predict sentiment based on the lemmatized review texts
      - then, proposes the top 5 products using with highest % positive sentiments
    '''
    # get lemmatized text
    df['lemma_review'] = df['reviews_text'].apply(lambda x: get_lemmatized(x))

    # predict the sentiment for the selected products by first extracting the features from the lemmatized review
    df['pred_sent'] = sentiment_model.predict(tfidf_transformer.transform(df['lemma_review']))

    # Now get the % positive reviews for each of these products
    df_positive = df[df.pred_sent == 1].groupby(['name','brand','manufacturer'])['lemma_review'].count().reset_index()
    df_all = df.groupby(['name','brand','manufacturer'])['lemma_review'].count().reset_index()

    df_positive.rename(columns = {'lemma_review': 'positive_reviews'}, inplace=True)
    df_all.rename(columns = {'lemma_review': 'total_reviews'}, inplace=True)

    df3_summary = pd.merge(df_positive,df_all)
    df3_summary['% positive'] = df3_summary.apply(lambda x: round(100 * x['positive_reviews']/x['total_reviews']) , axis=1)
    df3_summary.sort_values(by=['% positive','total_reviews'], ascending=False, inplace=True)
    print(df3_summary.head())

    return df3_summary.head()
