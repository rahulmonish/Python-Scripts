from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataframe= pd.read_csv(r'C:\Users\u65988\Documents\expression_clustter_2.csv')
#dataframe["Cluster_Center"] = np.nan
#dataframe = dataframe.fillna('0')
dataframe= dataframe['Execution Block Name'].dropna(axis=0, how= 'any')
#Name_String = dataframe['Name String']
#Name_String_Dict = {}
string=""
for i in dataframe:
    string += i


#s = ["A","B","A","C","B","A","B","A","A","B","B","C","B","A","B","A"]
s= "start ifstart elsepost"


d={}
MINLEN =len(s)+1
MINCNT =10
for sublen in range(MINLEN,int(len(s)/MINCNT)):
    for i in range(0,len(s)-sublen):
        sub = s[i:i+sublen]
        print(s)
        cnt = s.count(sub)
        if cnt >= MINCNT and sub not in d:
             d[sub] = cnt
