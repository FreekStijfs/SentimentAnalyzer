from itertools import chain
import pandas as pd
import json
import time
import pprint

# setting input and output path variables
input_tweets = "geotagged_tweets.jsons"
out = "tweets_users_places.csv"

# initializing features we will extract from dataset
tweet_features = ['id', 'text', 'created_at', 'retweet_count', 'favorite_count', 'lang', 'timestamp_ms']
user_features = ['location', 'protected', 'verified', 'followers_count', 'friends_count', 'listed_count', 'favourites_count',\
    'statuses_count', 'created_at', 'lang']
place_features = ['full_name', 'name', 'place_type', 'country_code', 'country', 'bounding_box']
columns = list(map(lambda s: 'tweet.' + s, tweet_features)) + list(map(lambda s: 'user.' + s, user_features)) +\
    list(map(lambda s: 'place.' + s, place_features))


def importer(input_tweets):
    try:
        start = time.time()
        # Loading or Opening the json file
        with open(input_tweets) as file:
            for line in file:
                # decode line from json to dict
                json_line = json.loads(line)
                # append list of selected features to lists
                tweets.append([json_line[key] for key in tweet_features])
                users.append(
                    [json_line['user'][key] for key in user_features] if json_line['user'] is not None else 'und')
                places.append(
                    [json_line['place'][key] for key in place_features] if json_line['place'] is not None else 'und')

                # status progress indicator
                if (len(tweets) % 25000 == 0):
                    print('Tweets loaded: {} in {} seconds'.format(len(tweets), time.time() - start))

            # merge lists in a row-wise fashion and put in dataframe
            data = [list(chain.from_iterable(x)) for x in zip(tweets, users, places)]
            df = pd.DataFrame(data=data, columns=columns)
            # save dataframe to csv
            df.to_csv(out)
        return df

    except Exception as e:
        pprint.pprint(json_line)
        print(e)

# initializing empty lists
tweets, users, places = list(), list(), list()
# running main importer
df = importer(input_tweets)
df.head()