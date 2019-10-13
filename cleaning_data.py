import pandas as pd
import re

df = pd.read_excel("datafile/TajinderBagga_friends_tweets_bio.xlsx")

tweets = [eval(i) for i in df["tweets"].tolist()]
bio = df["bio"].tolist()
print(tweets[3])
print(len(bio))
clean_tweets=[]
clean_bio=[]
for i in bio:
	try:
		n_bio = re.sub('[^a-zA-Z0-9/#@]+',' ', i).strip()
		if n_bio=="":
			clean_bio.append("NO_BIO")
		else:
			clean_bio.append(n_bio)	
	except Exception as e:
		clean_bio.append("NO_BIO")
for i in tweets:
	clean_tweets.append([re.sub(r'http\S+RT', '', j).strip() for j in i])

print(clean_tweets[3])	
tweets = clean_tweets
clean_tweets=[]
for i in tweets:
	clean_tweets.append([re.sub('[^a-zA-Z0-9/#@]+',' ',j).strip() for j in i if re.sub('[^a-zA-Z0-9/#@]+',' ',j).strip()!=""])



print(len(clean_tweets), len(clean_bio))
print(clean_tweets[3], clean_bio[3], len(clean_tweets[3]))

df2 = pd.DataFrame()
df2["friends_id"] = df["friends_id"]
df2["tweets"] = clean_tweets
df2["bio"] = clean_bio
# df2.to_excel("datafile/TajinderBagga_friends_tweets_bio_cleaned.xlsx")
