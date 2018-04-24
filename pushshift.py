import requests
import ujson as json
import re
import time
import datetime


PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%d-%m %H:%M:%S')



debug = open("debug.txt", "w")


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
    max_created_utc = 0
    max_id = 0
    file = open("data.json","w")
    counter = 1

    print("started collecting at epoch time:",ts,"which is:",st,file=debug)
    while 1:
        nothing_processed = True
        objects = fetchObjects(**kwargs,after=max_created_utc,before=1524507927)    # can also use "before="
        if objects is None:
            print("weired none object 1",file=debug)
            continue
            break
        for object in objects:    # runs 1000 times (size times)
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

                #TODO: improve filtering up ahead
                if object['author_flair_text'] != None and object['author_flair_text'].isdigit():
                    print(json.dumps(object,sort_keys=True,ensure_ascii=True),file=file)
        # TODO: replace by some not-hard-coded utc date
        if created_utc >= 1524507927:
            print("break due to date limit - reached today",file=debug)
            break

        if (file.tell()) > 4026531840:
            print("break due to size limit - reached 30 GB",file=debug)
            break

        if nothing_processed:
            return
        max_created_utc -= 1
        time.sleep(.5)
        counter -= 1

        print(humansize(file.tell()))  # debug tracking after file size

    file.close()


# small byte to mb gb etc convertor
suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


process(subreddit="teenagers",type="submission")


debug.close()