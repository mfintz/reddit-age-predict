import codecs
from json import JSONDecoder, JSONDecodeError
import requests
# import ujson as json
import re
import time
import datetime
import json
import threading
import os
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%d-%m %H:%M:%S')




# small byte to mb gb etc convertor
suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])



PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"

# gets a filename as a string, the file should contain ALL the names and ages (sorted by age)
def handleAuthorInput(filename):
    file = open(filename, "r")
    thirteen = []
    fourteen = []
    fifteen = []
    sixteen = []
    seventeen = []
    eighteen = []
    nineteen = []
    old = []
    for index, line in enumerate(file):
        if line[0:2] == "13":
            thirteen.append(line[4:-1])
        if line[0:2] == "14":
            fourteen.append(line[4:-1])
        if line[0:2] == "15":
            fifteen.append(line[4:-1])
        if line[0:2] == "16":
            sixteen.append(line[4:-1])
        if line[0:2] == "17":
            seventeen.append(line[4:-1])
        if line[0:2] == "18":
            eighteen.append(line[4:-1])
        if line[0:2] == "19":
            nineteen.append(line[4:-1])
            if index == 2185:
                print()
        if line[0:3] == "OLD":
            old.append(line[5:-1])

    all_ages = []
    all_ages.append(thirteen)
    all_ages.append(fourteen)
    all_ages.append(fifteen)
    all_ages.append(sixteen)
    all_ages.append(seventeen)
    all_ages.append(eighteen)
    all_ages.append(nineteen)
    all_ages.append(old)

    for index, age in enumerate(all_ages):
        cur_file = open(str(13 + index) + ".txt", "w")
        print("There are ",len(all_ages[index]), " users in age ", (13+index))
        for author in age:
            print(author, file=cur_file)
        cur_file.close()

    file.close()
    return all_ages


# all_ages = handleAuthorInput("2017authorsCommentsSubmissions.txt")


# using PushShift, there's only 2 endpoints - submissions and comments - so we'll do it separately
# example of a search : all submissions in 2017 for certain author
# https://api.pushshift.io/reddit/search/submission/?author=Itwentrightin&sort=asc&size=1000&after=1483221600&before=1514757600


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



debug = open("redditdebug.txt","w")


def process(age,**kwargs):
    print("$$$$$$$$$$$$$$$$$$$$   ",age,"   $$$$$$$$$$")
    if not os.path.exists("\\jsons\\"):
        os.makedirs("\\jsons\\")

    last_year_utc = 1483221600
    max_created_utc = 1483221600    # 1483221600 is 1.1.2017 - since we want to be as sure as we can with flairs
    max_id = 0

    # Json file will eventually contain all comments or all submissions of a certain age # NOTICE : doing append, so if wants to start over, should clean files before
    json_file = open("\\jsons\\"+ str(age)+str(kwargs['type'])+".json","a")
    corpus_file = codecs.open("\\jsons\\" + (str(age)+str(kwargs['type']))+".txt","a")
    backup = open("backupReddit.txt","a")
    counter = 1

    print("started collecting posts by author at epoch time:",ts,"which is:",st,file=debug)
    while 1:
        nothing_processed = True
        objects = fetchObjects(**kwargs,after=max_created_utc,before=1514757600)    # can also use "before=" . 1514764861 is 1.1.2018
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


                # if object['author'] == "ABlankNam3dKid":
                #     print()
                #     if obj_index == 161:
                #         print()


                if object['author'] == '[deleted]':
                    continue
                print(json.dumps(object,sort_keys=True,ensure_ascii=True),file=json_file)   # print to json # consider returning the ,indent=4 to arguments

                # if kwargs['type'] == "comment":
                #     length = len(object['body'])
                #     if length > 255:
                #         try:
                #             print(object['body'].encode("utf-8"),file=corpus_file)
                #         except Exception as e:
                #             continue
                #     else:
                #         print(object['body'], file=corpus_file)
                # if kwargs['type'] == "submission":
                #     if 'selftext' in object:
                #         length = len(object['selftext'])
                #         if length > 0:
                #             if length > 255:
                #                 try:
                #                     print(object['selftext'].encode("utf-8"), file=corpus_file)
                #                 except Exception as e:
                #                     continue
                #             else:
                #                 print(object['selftext'],file=corpus_file)
                #
                #             # print(object['selftext'].encode("utf-8"),file=corpus_file)
                #             #TODO: optional
                #             # print(object['title'],file=corpus_file)





        if nothing_processed:
            break
        max_created_utc -= 1
        # time.sleep(.5)
        counter -= 1

    backup.close()
    json_file.close()
    corpus_file.close()

    return all_ages




# retrieving all posts (comments and submissions) of all users in different ages
def handleAges(all_ages):
    for index, age in enumerate(all_ages):
        for author in age:
            all_ages_submissions = process((13 + index), type="submission", author=author, size=1000)
            all_ages_comments = process((13 + index), type="comment", author=author, size=1000)




# handleAges(all_ages)





#TODO:
#   text = text.replace(u'\ufeff', '')  # Char fix for some of the files



# gets a directory , runs through all it's json files and making new text file with just the body
def handleJson(input_path):
    dir_files = os.listdir(input_path)


    for file in dir_files:
        sents = codecs.open(input_path + "\\" + file[:-4] + "txt", 'w', "utf-8")
        # sents = codecs.open("19sentSubmission.txt", "w", "utf-8")
        js = open(input_path + "\\" + file, 'r')
        for line in js:
            item = json.loads(line)
            # Submissions
            if 'selftext' in item:
                if item['selftext'] != '[removed]' and len(item['selftext'])>0:
                    print(item['selftext'].replace("\n",""), file=sents)
                    # sents.write(item['selftext'])
            # Comments
            elif 'body' in item:
                if item['body'] != '[removed]' and len(item['body'])>0:
                    print(item['body'].replace("\n",""), file=sents)

        sents.close()



# gets the json directory name
# handleJson("jsons")



WHITE_SPACE = [' ', '\n', '\r', '\t']


def splitSent(input_path):
    dir_files = os.listdir(input_path)


    #TODO: also , take care of skipping the folder - it is considered as a file now
    if not os.path.exists(input_path + "\\splitsents\\"):
        os.makedirs(input_path + "\\splitsents\\")
    for file in dir_files:
        if file.endswith(".txt"):
            # new file
            sents = codecs.open(input_path + "\\splitsents\\" + file + ".splitsentences", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r',"utf-8")
            # for line in src:
            #     print()
            #
            # sents.close()

            sentences = []
            final_text = ""
            for line in src:
                final_text += line

            i = 0
            for j in range(len(final_text)):
                if i < j:
                    # any line break (we entered) ends a sentence
                    if final_text[j] in ['\r', '\n']:
                        sentences.append(final_text[i:j])
                        i = j + 1
                    # look for characters which signify end of sentence
                    elif final_text[j] in ['.', ';', '?', '!']:
                        # a '.'
                        if final_text[j] == '.':
                            k = j + 1
                            # a '.' with whitespace afterwards
                            if (k < len(final_text)) and (final_text[k] in WHITE_SPACE):
                                sentences.append(final_text[i:k])
                                i = k + 1
                            # a '.' with quotes afterwards
                            elif (k < len(final_text)) and (final_text[k] in ['"', '״', '\'']):
                                if (k + 1 < len(final_text)) and (final_text[k + 1] in WHITE_SPACE):
                                    sentences.append(final_text[i:k + 1])
                                    i = k + 2
                            else:
                                # look for more '.'s
                                while (k < len(final_text)) and (final_text[k] == '.'):
                                    k += 1
                                # many '.'s. otherwise - a letter after a '.' - not the end of the sentence
                                if k>= len(final_text):
                                    print("debug")
                                if final_text[k] in WHITE_SPACE:
                                    sentences.append(final_text[i:k])
                                    i = k + 1
                        # a '?' or a '!'
                        elif final_text[j] in ['?', '!']:
                            k = j + 1
                            while (k < len(final_text)) and (final_text[k] in ['?', '!']):
                                k += 1
                            # a whitespace should appear after them
                            if final_text[k] in WHITE_SPACE:
                                sentences.append(final_text[i:k])
                                i = k + 1
                        # a ';'
                        else:
                            k = j + 1
                            if (k < len(final_text)) and (final_text[k] in WHITE_SPACE):
                                sentences.append(final_text[i:k])
                                i = k + 1
            first_sentence = True
            for sent in sentences:
                if first_sentence:
                    first_sentence = False
                else:
                    sents.write('\r\n')
                sents.write(sent)

        sents.close()


splitSent("jsons\\texts")

print()
debug.close()