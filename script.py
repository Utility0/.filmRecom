import pandas as pd
import numpy as np
from datetime import datetime



# Import Rating Dataset
path = 'data/ratings.csv'
rating = pd.read_csv(path, sep=',')
# Import Film Dataset
path = 'data/movies.csv'
film = pd.read_csv(path, sep=',')
# Extract Needed Data From The Data Frames
rating = rating[['userId','movieId','rating']]
film = film[['movieId','title']]
# Main Matrix
df = rating.pivot(index='userId', columns='movieId', values='rating')
df = df.fillna(0)
dfnp = (df.to_numpy())

P=pd.read_csv("data/P.csv", sep=',')
Q=pd.read_csv("data/Q.csv", sep=',')
P = P.set_index('index').to_numpy()
Q = Q.set_index('index').to_numpy()
prednp = np.dot(P,Q)
pred = pd.DataFrame(prednp)

def getFilms(userId,M,Pred,film):
    predid = Pred[Pred.index==userId]
    mid = M[M.index==userId]
    mid = abs(mid - 1)
    tp = (predid*mid).T
    tp = tp.sort_values(by = userId, ascending=False)
    listFilm = tp.head().index
    return film[film["movieId"].isin(list(listFilm))]
    

print(getFilms(1,df,pred,film))

# nonZero = list()
# I,J = df.shape
# for i in range(I):
#     for j in range(J):
#         if dfnp[i][j] > 0:
#             nonZero.append((i,j))

print(pred.shape)

# sum = 0
# for i in nonZero:
#     sum = pow(dfnp[i[0]][i[1]]-prednp[i[0]][i[1]],2)
# sum /= len(nonZero)
# sum = np.sqrt(sum)
# print(sum)

