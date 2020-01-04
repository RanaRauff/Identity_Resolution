import pandas as pd
import pickle
from sklearn.cluster import KMeans
import numpy as np
import collections
from rake_nltk import Rake
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
	
	w2v_model = word2vec.Word2Vec(filtered_sentence, size=feature_size,
    	                      window=window_context, min_count=min_word_count,
        	                  sample=sample, iter=50)

	return w2v_model


def kmeans_model(w2v_feature_array,k):
	kmean = KMeans(n_clusters=k, random_state=0).fit(w2v_feature_array)

	clusters = collections.defaultdict(list)
	for i, label in enumerate(kmean.labels_):
		clusters[label].append(i)
	return dict(clusters),kmean

def affinity_propagation(w2v_feature_array):

    ap = AffinityPropagation()
    ap.fit(w2v_feature_array)
    max_iter=200
    damping = 0.5
    affinity = "euclidean"
    clusters = collections.defaultdict(list)
    for i, label in enumerate(ap.labels_):
        clusters[label].append(i)
    return dict(clusters)

def kmean_distortions(w2v_feature_array,k):

    distortions = []
    for i in range(1, n+1):
        km = KMeans(
            n_clusters=k, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(w2v_feature_array)
        distortions.append(km.inertia_)

    plt.plot(range(1, n+1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def kmeans_center(w2v_feature_array,k):

    kmean = KMeans(n_clusters=k, random_state=0).fit(w2v_feature_array)

    centroids = kmean.cluster_centers_

    tsne_init = 'pca'  # could also be 'random'
    tsne_perplexity = 20.0
    tsne_early_exaggeration = 4.0
    tsne_learning_rate = 1000
    random_state = 1
    model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
                 early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

    transformed_centroids = model.fit_transform(centroids)
    print(transformed_centroids)
    plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
    plt.show()

def tagging(filename):

    df = pd.read_excel(filename)
    cols = df.columns.values
    r=Rake()
    df2=pd.DataFrame()
    for i in cols:
        # print(i)
        sent = [str(j) for j in df[i].values if j!=0]
        # print(sent)
        r.extract_keywords_from_text(" ".join(sent))
        # print(r.get_word_frequency_distribution())
        # print(r.get_word_degrees())
        fdis = r.get_word_frequency_distribution()
        wdig = r.get_word_degrees()
        fdis_ls=[]
        wdig_ls=[]
        wdig = {a: b for a, b in sorted(wdig.items(), key=lambda item: item[1], reverse=True)}
        # print(wdig)
        for j in fdis.most_common():
            # print(j[0])
            if len(j[0])>3:
                fdis_ls.append(j[0])
        for j in wdig.keys():
            if len(j) > 3:
                wdig_ls.append(j)
        print(fdis_ls[:5])
        print(wdig_ls[:5])
        res = [fdis_ls[:5],wdig_ls[:5]]
        df2[i] = res

        # break
    df2.to_excel("datafile/tagged.xlsx")

def friend_id_tagging(filename):
    df=pd.read_excel("datafile/TajinderBagga_friends_tweets_bio_cleaned.xlsx")
    df2 = pd.read_excel(filename)
    cols = df2.columns
    df3=pd.DataFrame()
    for k in cols:
        print(k)
        a = eval(df2[k][0])
        b = eval(df2[k][1])
        a.extend(b)
        attributes = set(a)
        print("attri: ", a)
        friends_id=[]
        for i, j in df.iterrows():
            # print(eval(j["tweets"]))
            if any(x in j["tweets"].lower() for x in attributes):
                # print(j["friends_id"])
                friends_id.append(j["friends_id"])
                # res.append(j["friends_id"])
                # twee.append(j["tweets"])
        df3[k] = friends_id+[0]*(10000-len(friends_id))
    df3.to_excel("datafile/friends_id_tagged.xlsx")




# filtered_sentence = tokenize()

# w2v_model = w2v(filtered_sentence)
# pickle.dump(w2v_model,open("latest_model.bin","wb"))
# w2v_model = pickle.load(open("latest_model.bin","rb"))


# w2v_feature_array = averaged_word_vectorizer(corpus=filtered_sentence, model=w2v_model,
#                                              num_features=100)

friend_id_tagging("datafile/tagged_50.xlsx")
n=30

# kmeans_center(w2v_feature_array,n)


# clusters,kmean = kmeans_model(w2v_feature_array, n)
# distortions=[]

# clusters = affinity_propagation(w2v_feature_array)



# ========================================================================================

clust=[]
clust_sent=[]
df2 = pd.DataFrame()

for cluster in range(n):
    print("cluster ",cluster,":")
    sent=[]
    clust.append("CLUSTER_NO "+str(cluster))
    for i,sentence in enumerate(clusters[cluster]):
        print("\tsentence ",i,": "," ".join(filtered_sentence[sentence]))
        sent.append(" ".join(filtered_sentence[sentence]))
    # clust_sent.append(sent)
    df2["CLUSTER_NO_"+str(cluster)] = sent+[0]*(10000-len(sent))
# df2["CLUSTER NUMBER"] = clust
# df2["CLUSTER"] = clust_sent
df2.to_excel("CLUSTER_affinity.xlsx")
# distortions=[]
# distortions.append(kmean.inertia_)


# ========================================================================================







# ap = AffinityPropagation()
# ap.fit(w2v_feature_array)
# cluster_labels = kmean.labels_
# for i,j in cluster_labels:
# 	print(i,j)
# 	break
# cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
#
# df=pd.concat([filtered_sentence, cluster_labels], axis=1)
# print(df.head())
# print(cluster_labels.head())
# pickle.dump(w2v_model,open("latest_model.bin","wb"))
# get document level embeddings
# w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,
#                                              num_features=feature_size)
# pd.DataFrame(w2v_feature_array)