# from twitterscraper import query_tweets
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import time
import tweepy
# form tweepy.streaming import StreamListener
# from tweepy import OAuthHandler
# from tweepy import StreamListener

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
				time.sleep(3*60)
	
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
		df.to_excel("FollowingOutput.xlsx")
	


if __name__ == '__main__':
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_KEY,ACCESS_SECRET)
	api = tweepy.API(auth)
	
	# try:
	user1=UserTwitter()
	user1.following("RahulRahul010")
	# except Exception as e:
	# 	print(f"ERROR || {e}")				
# user=api.friends(screen_name="pradip103")
# # print(user)
# c=0
# for i in user:
# 	c+=1
# 	print(f" {c} {i.name} {i.screen_name}")
# 	print("================================================")
# men=api.mentions_timeline()
# for i in men:
# 	print(str(i.id) + " - "+ i.text)
# print(os.getenv("API_KEY")) 
