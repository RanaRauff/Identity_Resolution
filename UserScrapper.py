# from twitterscraper import query_tweets

import os
import time
import tweepy
import logging
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# from tweepy.streaming import StreamListener
# from tweepy import OAuthHandler
# from tweepy import StreamListener

# ======================================================
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "ErrorLog.txt", level = logging.DEBUG, format = LOG_FORMAT)
logger = logging.getLogger()
# ======================================================




# ======================================================
CONSUMER_KEY = os.getenv("API_KEY")
CONSUMER_SECRET =os.getenv("API_SECRET_KEY")
ACCESS_KEY =os.getenv("ACCESS_TOKEN")
ACCESS_SECRET =os.getenv("ACCESS_TOKEN_SECRET")
# ======================================================



class UserTwitter():
	"""docstring for UserTwitter"""
	
	def limit_generator(self, cursor):
		while True:
			try:
				yield cursor.next()
			except tweepy.RateLimitError:
				time.sleep(15*60)
	
	def following(self, user_name):

		df=pd.DataFrame()
		name_ls=[]
		screen_name_ls=[]

		for foll in self.limit_generator(tweepy.Cursor(api.friends, screen_name=user_name).items()):		
			# print(f"{foll.name} || {foll.screen_name}")
			name_ls.append(foll.name)
			screen_name_ls.append(foll.screen_name)
		
		df["name"] = name_ls
		df["screen_name"] = screen_name_ls
		df.to_excel("datafile/FollowingOutput.xlsx")
	
	def friends_id(self, screen_name):
		
		id_ls = api.friends_ids(screen_name = screen_name)
		df = pd.DataFrame()
		df["friends_id"] = id_ls
		df.to_excel(f"datafile/{screen_name}.xlsx")


	def latest_tweets(self, id, limit):
		# api.get_user(id=2288047490)._json["screen_name"]
		tweets_ls=[]
		try:
			userdata = api.get_user(id=id)
			print(f"======================{userdata._json['name']} - {userdata._json['screen_name']}======================\n")
			for i in api.user_timeline(id=id,tweet_mode='extended', count = 20):
				print(i._json["full_text"])	
				print("----------------------------------------------------------------------------------")
				tweets_ls.append(i._json["full_text"])
			return tweets_ls
		except Exception as e:
			print(f"ERROR | User Not Found OR No TWEETS for user id = {id}")
			return tweets_ls		
		# print(api.friends_ids(screen_name = user))
		# ls=api.friends_ids(screen_name = user)
		# print(api.user_timeline(id=ls, count = 10))
		# for i in ls:
		# # 	# print()
		# 	for j in api.user_timeline(id=i, count = 10):
		# 		print(j.text)
		# 		print("=====================================================================================")
		# # 	# for j in api.get_user(id=i):
		# # 	# 	print(j)
		# # 	# 	print("==================================================")
		# 	break	

	def read_tweets_by_screen_name(self, screen_name):
		
		try:
			df = pd.read_excel(f"datafile/{screen_name}.xlsx")
			# print(df["friends_id"])
			friends_id_ls = df["friends_id"].tolist()
			# print(friends_id_ls)
			tweets_ls=[]
			for id in friends_id_ls:
				tweets_ls.append(self.latest_tweets(id, 20))
			df2 = pd.DataFrame()
			df2["friends_id"] = friends_id_ls
			df2["tweets"] = tweets_ls
			df2.to_excel(f"datafile/{screen_name}_friends_tweets.xlsx")	
		
		except Exception as e:
			print(f"ERROR: User Does Not Exist. ({e})")	 	

if __name__ == '__main__':

	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_KEY,ACCESS_SECRET)
	api = tweepy.API(auth)

	dirs = ["datafile"]
	for dir in dirs:
		if not os.path.exists(dir):
			os.makedirs(dir)
	
	try:
		user1=UserTwitter()
		# user1.friends_id("TajinderBagga") # Makes A list of friend id
		user1.read_tweets_by_screen_name("TajinderBagga") # saves the tweets of the targeted user in datafiles 
		# user1.following("TajinderBagga")
		# user1.latest_tweets(10,"TajinderBagga")
	except Exception as e:
		print(f"ERROR || {e}")
		logger.error("EXCEPTION ERROR IN MAIN", exc_info=True)				
# user=api.friends(screen_name="pradip103")
# # print(user)
# c=0
# RahulRahul010
# for i in user:
# 	c+=1
# 	print(f" {c} {i.name} {i.screen_name}")
# 	print("================================================")
# men=api.mentions_timeline()
# for i in men:
# 	print(str(i.id) + " - "+ i.text)
# print(os.getenv("API_KEY")) 
