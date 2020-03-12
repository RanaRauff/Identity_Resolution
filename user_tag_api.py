import sys
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import tweepy
import plotly.express as px

CONSUMER_KEY, CONSUMER_SECRET = "CONSUMER_KEY", "CONSUMER_SECRET"
ACCESS_KEY,ACCESS_SECRET="ACCESS_KEY","ACCESS_SECRET"


def radarChart(id,name,screen_name,location,df,df2):

    ls=[0]*len(df.columns)
    ind=0
    for i in df.columns:
        if id in df[i].values:
            print(f"This ID:{id} is present in {i}")
            ls[ind]+=1
        ind+=1    
    print(ls)
    fig = go.Figure(data=go.Scatterpolar(
  r=ls,
  theta=df2.iloc[0].values,
  fill='toself',
  name=name,
  
))

    
    fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True
    ),
  ),
  showlegend=False,
  title=go.layout.Title(text=f"Tagging for {name}(@{screen_name}) - ID: {id} - Location:{location}")
  
)
    fig.show()   
            

if __name__ == '__main__':
	file1,file2,id=sys.argv[1:]
	id=int(id)
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_KEY,ACCESS_SECRET)
	api = tweepy.API(auth)
	user=api.get_user(id=id)
	print(user._json)
	print(file1,file2,id)
	name,screen_name,location=user._json["name"],user._json["screen_name"],user._json["location"]
	print(name,screen_name,location)
	df = pd.read_excel(file1)
	df2 = pd.read_excel(file2)
	radarChart(id,name,screen_name,location,df,df2)