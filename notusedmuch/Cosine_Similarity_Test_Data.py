from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\dbmodules\test_data.csv')
dataframe["Cluster_Center"] = np.nan
dataframe = dataframe.fillna('0')

Name_String = dataframe['Name String']
Name_String_Dict = {}

for i in range(0,len(Name_String)):
    Name_String_Dict[Name_String[i]]=[]
    for j in range(i+1,len(Name_String)):
        expressions = (Name_String[i],Name_String[j])
        tfidf_vectorizer=TfidfVectorizer(analyzer="char")
        tfidf_matrix=tfidf_vectorizer.fit_transform(expressions)
        cs=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
        Name_String_Dict[Name_String[i]].append(cs)


expressions = ('Rahul','rah')
tfidf_vectorizer=TfidfVectorizer(analyzer="char")
tfidf_matrix=tfidf_vectorizer.fit_transform(expressions)
cs=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
print(cs)