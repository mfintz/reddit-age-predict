###############
# NLP LAB 	  #
# Matan Fintz #
###############

import prawcore
import requests
# import ujson as json
import re
import time
import datetime
import json
prawcore.exceptions.NotFound
import threading


import praw
import codecs

# PRAW's reddit object for more info
# needs to enter your client id,secret and password
reddit = praw.Reddit(client_id='',
                     client_secret='', password='',
                     user_agent='', username='')







PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%d-%m %H:%M:%S')



debug = open("debug.txt", "w")


# small byte to mb gb etc convertor
suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])




def fetchObjects(**kwargs):
    # Default params values
    params = {"sort_type":"created_utc","sort":"asc","size":1000}
    for key,value in kwargs.items():
        params[key] = value
    print(params)
    type = "comment"
    if 'type' in kwargs and kwargs['type'].lower() == "submission":
        type = "submission"
    r = requests.get(PUSHSHIFT_REDDIT_URL + "/" + type + "/search/",params=params)
    if r.status_code == 200:
        response = json.loads(r.text)
        data = response['data']
        sorted_data_by__id = sorted(data, key=lambda x: int(x['id'],36))
        return sorted_data_by__id




def process(**kwargs):
    last_year_utc = 1483221600
    max_created_utc = 1483221600    # 1483221600 is 1.1.2017 - since we want to be as sure as we can with flairs
    max_id = 0
    file = open("data.json","w")
    backup = open("backup.txt","a")
    counter = 1
    thirteen = set()
    fourteen = set()
    fifteen = set()
    sixteen = set()
    seventeen = set()
    eighteen = set()
    nineteen = set()
    old = set()
    print("started collecting at epoch time:",ts,"which is:",st,file=debug)
    while 1:
        nothing_processed = True
        objects = fetchObjects(**kwargs,after=max_created_utc,before=1514757600)    # can also use "before=" . 1514757600 is 1.1.2018
        if objects is None:
            print("weired none object 1",file=debug)
            continue
        for obj_index,object in enumerate(objects):    # runs 1000 times (size times)
            if object is None:
                print("weired none object 2", file=debug)
                break
            id = int(object['id'],36)
            if id > max_id:
                nothing_processed = False
                created_utc = object['created_utc']
                max_id = id
                if created_utc > max_created_utc:
                    max_created_utc = created_utc

                print("object number:", obj_index)

                author = object['author_flair_text']
                # filtering - just the last year's users
                if author != None:
                    # print(json.dumps(object,sort_keys=True,ensure_ascii=True,indent=4),file=file)   # print to json
                    if object['author'] == '[deleted]':
                        continue
                    # check if already seen this user
                    if author in thirteen or author in fourteen or author in fifteen or author in sixteen or author in seventeen or author in eighteen or author in nineteen or author in old:
                        continue
                    try:
                        user_creation_date = reddit.redditor(object['author']).created_utc
                    except Exception as e:
                        print("reached a user who deleted his profile - can't verify his creation date",file=debug)
                        continue

                    if user_creation_date >= last_year_utc:
                        if object['author_flair_text'] == "13":
                            thirteen.add(object['author'])
                            print("13,",object['author'],file=backup)
                        if object['author_flair_text'] == "14":
                            fourteen.add(object['author'])
                            print("14,",object['author'],file=backup)
                        if object['author_flair_text'] == "15":
                            fifteen.add(object['author'])
                            print("15,",object['author'],file=backup)
                        if object['author_flair_text'] == "16":
                            sixteen.add(object['author'])
                            print("16,", object['author'], file=backup)
                        if object['author_flair_text'] == "17":
                            seventeen.add(object['author'])
                            print("17,", object['author'], file=backup)
                        if object['author_flair_text'] == "18":
                            eighteen.add(object['author'])
                            print("18,", object['author'], file=backup)
                        if object['author_flair_text'] == "19":
                            nineteen.add(object['author'])
                            print("19,", object['author'], file=backup)
                        if object['author_flair_text'] == "OLD":
                            old.add(object['author'])
                            print("old,", object['author'], file=backup)


        # for the slight chance there's too many users
        if len(thirteen) > 200000 and len(fourteen) > 200000 and len(fifteen) > 200000 and len(sixteen) > 200000 and len(seventeen) > 200000 and len(eighteen) > 200000 and len(nineteen) > 200000 and len(old) > 200000:
            break


        if nothing_processed:
            #
            break
        max_created_utc -= 1
        # time.sleep(.5)
        counter -= 1

        print(humansize(file.tell()))  # debug tracking after file size
    backup.close()
    file.close()

    all_ages = []
    all_ages.append(thirteen)
    all_ages.append(fourteen)
    all_ages.append(fifteen)
    all_ages.append(sixteen)
    all_ages.append(seventeen)
    all_ages.append(eighteen)
    all_ages.append(nineteen)
    all_ages.append(old)

    return all_ages



all_ages_comments  = []
all_ages_submissions = process(subreddit="teenagers",type="submission")
all_ages_comments = process(subreddit="teenagers",type="comment")


# manipulate this loop according to the needs
for index, age in enumerate(all_ages_submissions):
    cur_file = open(str(13 + index) + ".txt", "w")
    combined_age = list(all_ages_comments[index].union(all_ages_submissions[index]))

    for name in combined_age:
        print(name,file=cur_file)
    cur_file.close()



debug.close()