import praw
import codecs



# Make a new file
output = codecs.open("prawOut.txt", 'w', "utf-8")


reddit = praw.Reddit(client_id='E8sXdgKwMzysNQ',
                     client_secret='Bfixm1o5wBTV8hPX8Pc5HVNN7Qg', password='nlpredditproject',
                     user_agent='Nlproject', username='nlpredditproject')

# https://www.reddit.com/user/Aperson3334/about.json
check = reddit.redditor("Aperson3334")
subreddit = reddit.subreddit('teenagers')
top_subreddit = subreddit.new(limit = 9999)
for index,submission in enumerate(top_subreddit):
    output.write(str(index))
    output.write(":")
    output.write(submission.title)
    output.write("\n")


output.close()
