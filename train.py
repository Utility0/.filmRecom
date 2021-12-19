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
df.head()


def matrix_factorization(M,F,steps,alpha,beta):
    index = M.index
    colnames = M.columns
    nuser,nfilm = M.shape
    M = M.to_numpy()
    P = np.random.rand(nuser,F)
    Q = np.random.rand(F,nfilm)
    nonZeroList = [(i,j) for i in range(nuser) for j in range(nfilm) if M[i][j] > 0]
    print('Acceptable error : '+str(0.01*len(nonZeroList)))
    for step in range(steps):
        for a in nonZeroList:
            i,j = a
            eij = M[i][j] - np.dot(P[i,:],Q[:,j])
            for f in range(F):
                P[i][f]=P[i][f] + alpha * (2 * eij * Q[f][j] - beta * P[i][f])
                Q[f][j]=Q[f][j] + alpha * (2 * eij * P[i][f] - beta * Q[f][j])
        eR = np.dot(P,Q)
        e = 0
        for a in nonZeroList:
            i,j = a
            e = e + pow(M[i][j]-np.dot(P[i,:],Q[:,j]),2)
            for f in range(F):
                e = e + (beta/2)*(pow(P[i][f],2)+pow(Q[f][j],2))
        if e < 0.01*len(nonZeroList):
            break
        if step %1==0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(str(step)+' - '+current_time+' - '+str(e))
    return P,Q,e
                        
        
P,Q,e = matrix_factorization(df,20,2000,0.005,0.02)

pd.DataFrame(P).to_csv("data/P.csv")
pd.DataFrame(Q).to_csv("data/Q.csv")
print(e)