import pandas as pd
import pickle
from sklearn.cluster import KMeans
import numpy as np
import collections
from datetime import datetime
from gensim.models import word2vec
from sklearn.cluster import AffinityPropagation
# from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
stopwords =  set(stopwords.words('english'))



def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector
    
   
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


def tokenize():
	df = pd.read_excel("datafile/TajinderBagga_friends_tweets_bio_cleaned.xlsx")
	# tweets = [eval(i) for i in df["tweets"].tolist()]
	# print(tweets[0])
	c=0
	cc=0
	ccc=0
	ls_data=[]
	for i,j in df.iterrows():
		# if eval(j["tweets"])==[]:
		# 	c+=1
		# if j["bio"]=="NO_BIO":
		# 	cc+=1
		t= eval(j["tweets"])
		b = j["bio"]
		if t == [] and b=="NO_BIO" or b=="NO BIO":
			ls_data.append([])
		else:
			if b!="NO_BIO" or b!="NO BIO":
				t.append(b)
			ls_data.append(t)	
						

	sentence = [[j.split(" ") for j in i] for i in ls_data]
	sentence_data=[]
	for i in sentence:
		sentence_data.append([[k.lower() for k in j if k not in stopwords] for j in i])
	
	# return sentence_data
	filtered_sentence = []
	for i in sentence_data:
		for j in i:
			filtered_sentence.append(j)
	print(len(filtered_sentence))
	return filtered_sentence		

		

def w2v(filtered_sentence):
	
	feature_size = 100    # Word vector dimensionality  
	window_context = 30          # Context window size                                                                                    
	min_word_count = 1   # Minimum word count                        
	sample = 1e-3   # Downsample setting for frequent words
	
	w2v = word2vec.Word2Vec(filtered_sentence, size=feature_size, 
    	                      window=window_context, min_count=min_word_count,
        	                  sample=sample, iter=50)

	return w2v_model


def kmeans_model(w2v_feature_array,k):
	kmean = KMeans(n_clusters=k, random_state=0).fit(w2v_feature_array)

	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmean.labels_):
		clusters[label].append(i)
	return dict(clusters)



filtered_sentence = tokenize()

# w2v_model = w2v(filtered_sentence)
w2v_model = pickle.load(open("latest_model.bin","rb"))


w2v_feature_array = averaged_word_vectorizer(corpus=filtered_sentence, model=w2v_model,
                                             num_features=100)

n=150
clusters = kmeans_model(w2v_feature_array, n)
for cluster in range(n):
    print("cluster ",cluster,":")
    for i,sentence in enumerate(clusters[cluster]):
        print("\tsentence ",i,": "," ".join(filtered_sentence[sentence]))
# ap = AffinityPropagation()
# ap.fit(w2v_feature_array)
# cluster_labels = kmean.labels_
# for i,j in cluster_labels:
# 	print(i,j)
# 	break
# cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])

# df=pd.concat([filtered_sentence, cluster_labels], axis=1)
# print(df.head())
# print(cluster_labels.head())
# pickle.dump(w2v_model,open("latest_model.bin","wb"))
# get document level embeddings
# w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,
#                                              num_features=feature_size)
# pd.DataFrame(w2v_feature_array)