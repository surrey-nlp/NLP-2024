from flask import Flask, render_template, request, redirect, url_for
from joblib import load

import pandas as pd
pd.set_option('display.max_colwidth', 1000)


class Tweet:
    def __init__(self, created_at, id, text):
        self.created_at = created_at
        self.id = id
        self.text = text


def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 50
    
    for tweet in [Tweet('2021', 1, 'it\'s unbelievable that in the 21st century we\'d need something like this. again. #neverump  #xenophobia '), 
                  Tweet('2020', 2, 'product of the day: happy man #wine tool  who\'s   it\'s the #weekend? time to open up &amp; drink up!'), 
                  Tweet('2020', 2, '@user i\'m not interested in a #linguistics that doesn\'t address #race &amp; . racism is about #power. #raciolinguistics brings'), 
                  Tweet('2020', 2, '@user why not @user mocked obama for being black.  @user @user @user @user #brexit'), 
                  Tweet('2020', 2, 'yeah! new buttons in the mail for me they are so pretty! :) #jewelrymaking #buttons')
                 ]:
        print(tweet.text)
        # Adding to list that contains all tweets
        tweets_list.append({'created_at': tweet.created_at,
                            'tweet_id': tweet.id,
                            'tweet_text': tweet.text})
    return pd.DataFrame.from_dict(tweets_list)


pipeline = load("text_classification.joblib")


def requestResults(kw):
    tweets = get_related_tweets(kw)
    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return data + str(tweets)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        kw = request.form['search']
        return redirect(url_for('success', kw=kw))


@app.route('/success/<kw>')
def success(kw):
    return "<xmp>" + str(requestResults(kw)) + " </xmp> "


if __name__ == '__main__' :
    app.run(debug=True)