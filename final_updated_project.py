#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd


# In[72]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[73]:


movies.head(1)


# In[74]:


credits.head(1)


# In[75]:


movies =movies.merge(credits,on='title')


# In[76]:


movies.head()


# In[77]:


# genre
# id
# keywords
# title
# overview
# cast
# crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[78]:


movies.head()


# In[79]:


movies.isnull().sum()


# In[80]:


movies.dropna(inplace = True)


# In[81]:


movies.duplicated().sum()


# In[82]:


movies.iloc[0].genres


# In[83]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','Fantasy','Scify']


# In[84]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[85]:


def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[86]:


movies['genres'] = movies['genres'].apply(convert)


# In[87]:


movies.head()


# In[88]:


movies['keywords']=movies['keywords'].apply(convert)


# In[89]:


movies.head()


# In[90]:


def convert3 (obj):
    L =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3 :
            L.append(i['name'])
            counter +=1
        else:
            break
    return L


# In[91]:


movies['cast'] =movies['cast'].apply(convert3)


# In[92]:


movies.head()


# In[93]:


def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[94]:


movies['crew'] =movies['crew'].apply(fetch_director)


# In[95]:


movies.head()


# In[96]:


movies['overview'][0]


# In[97]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[98]:


movies.head()


# In[ ]:





# In[99]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" " ,"")for i in x])


# In[100]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])


# In[101]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])


# In[102]:


movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[103]:


movies.head()


# In[104]:


movies.head()


# In[105]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[106]:


movies.head()


# In[107]:


new_df = movies[['movie_id','title','tags']]


# In[108]:


new_df


# In[109]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[110]:


new_df.head()


# In[111]:


new_df['tags'][0]


# In[112]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[113]:


new_df.head()


# In[114]:


get_ipython().system('pip install nltk')


# In[115]:


import nltk


# In[116]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[117]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
        


# In[118]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[119]:


new_df['tags'][0]


# In[120]:


new_df['tags'][1]


# In[121]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words ='english')


# In[122]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[123]:


vectors


# In[124]:


vectors[0]


# In[125]:


cv.get_feature_names_out()


# In[126]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[127]:


from sklearn.metrics.pairwise import cosine_similarity


# In[128]:


similarity = cosine_similarity(vectors)


# In[129]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[130]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i)
        


# In[131]:


recommend('Batman Begins')


# In[132]:


new_df.iloc[1216].title


# In[133]:


import pickle


# In[134]:


pickle.dump(new_df,open('movies.pickle','wb'))


# In[135]:


new_df['title'].values


# In[136]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[137]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




