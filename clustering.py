import pandas as pd
from datetime import datetime
from gensim.models import word2vec

# feature_size = 100    # Word vector dimensionality  
# window_context = 30          # Context window size                                                                                    
# min_word_count = 1   # Minimum word count                        
# sample = 1e-3   # Downsample setting for frequent words




# tokenized_corpus = [ for i in df["tweets"]]
# print(tokenized_corpus[:20])

# w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, 
#                           window=window_context, min_count=min_word_count,
#                           sample=sample, iter=50)



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
	print(tweets[0])
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
		if t == [] and =="NO_BIO":
			ls_data.append([])
		else:
						
	print(c,cc,ccc)



print(tokenize())
# get document level embeddings
# w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,
#                                              num_features=feature_size)
# pd.DataFrame(w2v_feature_array)