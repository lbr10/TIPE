import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
plt.style.use('seaborn-whitegrid')

def graph(n) :
    return np.zeros((n,n))

G = np.array([[  0.,   5.,  np.infty,  np.infty,  np.infty,  np.infty,  np.infty,  np.infty],
       [  5.,   0.,   0.,   0.,  np.infty,  np.infty,  np.infty,  np.infty],
       [ np.infty,   0.,   0.,  np.infty,   4.,   0.,  np.infty,  np.infty],
       [ np.infty,   0.,  np.infty,   0.,  np.infty,  np.infty,  np.infty,   8.],
       [ np.infty,  np.infty,   4.,  np.infty,   0.,  np.infty,  np.infty,  np.infty],
       [ np.infty,  np.infty,   0.,  np.infty,  np.infty,   0.,   0.,  np.infty],
       [ np.infty,  np.infty,  np.infty,  np.infty,  np.infty,   0.,   0.,   0.],
       [ np.infty,  np.infty,  np.infty,   8.,  np.infty,  np.infty,   0.,   0.]])

def floyd_warshall(G) :
    W = deepcopy(G)
    n = len(W)
    for k in range(n) :
        for i in range(n) :
            for j in range(n) :
                W[i,j] = min(round(W[i,j],3),round(W[i,k] + W[k,j],3))
    return W

def mat2sa(G) :
    n = len(G)
    S = [k for k in range(n)]
    A = []
    for i in range(n) :
        for j in range(i + 1, n) :
            if G[i][j] != np.infty :
                A.append((i,j,G[i][j]))
    return (S,A)

        
def sa2mat(G) :
    (S,A) = G
    n = len(S)
    M = np.array([[np.infty for _ in range(n)] for _ in range(n)])
    for a in A :
        (i,j,d) = a
        M[pos_list(S,i)][pos_list(S,j)] = d
        M[pos_list(S,j)][pos_list(S,i)] = d
    for i in range(n) :
        M[i][i] = 0.
    return M

def inter(a,b,c,d) :
    (xa,ya) = a
    (xb,yb) = b
    (xc,yc) = c
    (xd,yd) = d
    m = (yb - ya) / (xb - xa)
    n = (yd - yc) / (xd - xc)
    p = ya - m * xa
    q = yc - n * xc
    if m == n and p == q :
        return True
    else :
        xinter = (q - p) / (m - n)
        yinter = m * xinter + p
        if min(xa,xb) < xinter < max(xa,xb) and min(ya,yb) < yinter < max(ya,yb) and min(xc,xd) < xinter < max(xc,xd) and min(yc,yd) < yinter < max(yc,yd) :
            return True
        else :
            return False

def rd_pt() :
    return (1000 * round(np.random.random(),3),1000 * round(np.random.random(),3))

def carte(n) :
    S = [rd_pt()]
    A = []
    for k in range(1,n) :
        print(k)
        s = rd_pt()
        if k == 1 :
            A.append((s,S[0]))
            print(A[0])
        else :
            for u in S :
                bool = True
                for a in A :
                    (x,y) = a
                    if inter(s,u,x,y) :
                        bool = False
                if bool :
                    A.append((s,u))
        S.append(s)
    draw_map(S,A)
    B = []
    for a in A :
        (x,y) = a
        (x1,y1) = x
        (x2,y2) = y
        M = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        B.append((x,y,M))
        
    return S,B

def draw_map(S,A) :
    for s in S :
        (x,y) = s
        plt.scatter(x,y)
    for a in A :
        (x,y) = a
        (x1,y1),(x2,y2) = x,y
        plt.plot([x1,x2],[y1,y2])
    plt.show()
    
def complexite_carte(n) :
    S = [rd_pt()]
    A = []
    for k in range(1,n) :
        s = rd_pt()
        if k == 1 :
            A.append((s,S[0]))
        else :
            for u in S :
                bool = True
                for a in A :
                    (x,y) = a
                    if inter(s,u,x,y) :
                        bool = False
                if bool :
                    A.append((s,u))
        S.append(s)
    return S,A

def test_complexite(f,n,p) :
    K = []
    T = []
    for k in range(2,n + 1) :
        print(k)
        K.append(k)
        t = time()
        for q in range(p) :
            X = f(k)
        T.append((time() - t)/p)
    plt.plot(K,[t ** (1/3) for t in T])
    plt.show()

def pos_list(L,x) :
    for i in range(len(L)) :
        if x == L[i] :
            return i
    return -1