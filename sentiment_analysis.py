import os                               # importing os - to help find the file path to open the reviews file
import spacy                            # importing spacy
nlp = spacy.load('en_core_web_sm')      # loading the language model
import pandas as pd                     # importing pandas
from textblob import TextBlob           # required to sentiment alaysis
import re                               # load the regex library to use to sanitise text

'''
This funtion takes in as input a `Review`.
The review text is then cleaned.
We then run sentiment alaysis and get a polarity score for the review.
(I have also outputted the polarity score of the uncleaned review to look for differences)
'''
def predict_sentiment(review):

    # Strip white space at the from and end, plus make the whole text lowercase
    cleaned = review.lower().strip() 
    # Eliminate the punctuation and other special characters
    cleaned = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", cleaned)
    # Remove the `Stop Words`
    doc = nlp(cleaned)
    filtered_tokens = [token for token in doc if not token.is_stop] 
    cleaned =  ' '.join([token.text for token in filtered_tokens])
    
    # Get the polarity score for the ORIGINAL uncleaned review
    blob_original = TextBlob(review)
    polarity_original = blob_original.sentiment.polarity
    # Get the polarity score for the CLEANED review
    blob_cleaned = TextBlob(cleaned)
    polarity_cleaned = blob_cleaned.sentiment.polarity

    # Print out the results in a clear manner
    print("--------------------------------------------------------------------------------------")
    print("ORIGINAL :: SCORE:", polarity_original, " -- ", review)
    print("CLEANED  :: SCORE:", polarity_cleaned, " -- ", cleaned)
    print("--------------------------------------------------------------------------------------\n")



# Use the OS library to get the correct file path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "amazon_product_reviews.csv")

# Open the file and read the reviews.text colomn into a dataframe
df = pd.read_csv(file_path,low_memory=False)['reviews.text']
# Remove all reviews which are blank
reviews = df.dropna()

# (We could loop through ALL the reviews using `for i in range(len(reviews))`, but the output would be too much for just testing.)
# Loop through 10 reviews and get the sentinment score.
for i in range(10):
    predict_sentiment(reviews[i])
