import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
plt.style.use('seaborn-whitegrid')


def Node(coords=(0,0),valley=0,roads=[],town=False,id=-1) :
    n = [coords,valley,roads,town,id]
    return n


def Road(start,end,height=0,is_tunnel=False) :
    r = [start,end,height,is_tunnel,longueur(start,end,height,is_tunnel)]
    return r


def longueur(start,end,h,is_tunnel) :
    '''Calcule approximativement la longueur d'une route allant de start à end et passant par un col de hauteur h'''
    if is_tunnel :
        (x1,y1) = start[0]
        (x2,y2) = end[0]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
    elif h == 0 :
        return 0
    else :
        (x1,y1) = start[0]
        (x2,y2) = end[0]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return np.sqrt(h**2 + L**2) + h * h * np.log((L + np.sqrt(h ** 2 + L ** 2)) / h) / L


def empty_net() :
    return []


def add_node(G,coord,valley,is_town,id) :
    '''Ajoute un noeud au réseau G données ses coordonnées, sa vallée et son statut (ville ou non)'''
    s = Node(coord,valley,[],is_town,id)
    G.append(s)


def add_road(G,start,end,height,is_tunnel=False) :
    '''Ajoute une route au réseau donnés son point de départ, son point d'arrivée et la hauteur du col les séparant.'''
    r = [start,end,height,is_tunnel,longueur(start,end,height,is_tunnel)]
    start[2].append(r)
    r = [end,start,height,is_tunnel,longueur(start,end,height,is_tunnel)]
    end[2].append(r)


def rem_road(G,start,end) :
    R = start[2]
    S = end[2]
    
    for r in R :
        if r[1] == end :
            R.remove(r)
            break
    
    for s in S :
        if s[2] == start :
            S.remove(s)
            break


def net2mat(N) :
    '''Associe à un réseau sa matrice d'adjacence'''
    n = len(N)
    M = np.array([[np.infty for _ in range(n)] for _ in range(n)])
    for i in range(len(N)) :
        for r in N[i][2] :
            j = r[1][4]
            M[i][j] = r[4]
            M[j][i] = r[4]
    for k in range(n) :
        M[k][k] = 0
    return M


def floyd_warshall(G) :
    '''Renvoie le distancier d'un graphe défini par sa matrice d'adjacence'''
    W = deepcopy(G)
    n = len(W)
    for k in range(n) :
        for i in range(n) :
            for j in range(n) :
                W[i,j] = min(W[i,j],W[i,k] + W[k,j])
    return W


def rd_pt(n) :
    '''Renvoie une liste de n points uniforméments répartis'''
    
    n -= 1
    p = int(np.ceil(np.sqrt((np.sqrt(2 * n + 1) + n + 1) / 2)) - 1)
    L = [(0,0)]

    for i in range(1,p + 1) :       #Liste des centres
        if n == 0 :
            break
        L.append((1, i - 1))
        n -= 1
        
        for k in range(1,i) :
            if n == 0 :
                break
            (x,y) = L[-1]
            L.append((x + 1, y - 1))
            n -= 1
            
        for k in range(i) :
            if n == 0 :
                break
            (x,y) = L[-1]
            L.append((x - 1, y - 1))
            n -= 1
            
        for k in range(i) :
            if n == 0 :
                break
            (x,y) = L[-1]
            L.append((x - 1, y + 1))
            n -= 1
            
        for k in range(i) :
            if n == 0 :
                break
            (x,y) = L[-1]
            L.append((x + 1, y + 1))
            n -= 1
    
    P = []
    for c in L :
        (x,y) = c
        P.append((5 * (np.random.random() - 0.5 + x), 5 * (np.random.random() - 0.5 + y)))
    
    return P


def inter(a,b,c,d) :
    (xa,ya) = a
    (xb,yb) = b
    (xc,yc) = c
    (xd,yd) = d
    if xb == xa :
        if xd == xc :
            if xd == xa :
                return True
            else :
                return False
        else :
            n = (yd - yc) / (xd - xc)
            q = yc - n * xc
            xinter = xa
            yinter = n * xa + q
            if min(xa,xb) < xinter < max(xa,xb) and min(ya,yb) < yinter < max(ya,yb) and min(xc,xd) < xinter < max(xc,xd) and min(yc,yd) < yinter < max(yc,yd) :
                return True
            else :
                return False
    else :
        if xd == xc :
            m = (yb - ya) / (xb - xa)
            p = ya - m * xa
            xinter = xc
            yinter = m * xinter + p
            if min(xa,xb) < xinter < max(xa,xb) and min(ya,yb) < yinter < max(ya,yb) and min(xc,xd) < xinter < max(xc,xd) and min(yc,yd) < yinter < max(yc,yd) :
                return True
            else :
                return False
        else :
            m = (yb - ya) / (xb - xa)
            n = (yd - yc) / (xd - xc)
            p = ya - m * xa
            q = yc - n * xc
            if m == n and p == q :
                return True
            elif m == n :
                return False
            else :
                xinter = (q - p) / (m - n)
                yinter = m * xinter + p
                if min(xa,xb) < xinter < max(xa,xb) and min(ya,yb) < yinter < max(ya,yb) and min(xc,xd) < xinter < max(xc,xd) and min(yc,yd) < yinter < max(yc,yd) :
                    return True
                else :
                    return False


def net_alea(n) :
    '''Crée un réseau aléatoire avec n noeuds'''
    N = empty_net()
    P = rd_pt(n)
    add_node(N,P[0],0,False,0)
    for k in range(1,n) :
        s = Node(P[k],k,[],False,k)
        if k == 1 :
            add_node(N,s[0],k,s[3],k)
            add_road(N,N[0],s,np.random.random() * 0.5)
        else :
            add_node(N,s[0],k,s[3],k)
            for u in N :
                if u[0] != s[0] :
                    bool = True
                    for x in N :
                        for r in x[2] :
                            y = r[1]
                            if x[4] < y[4] :
                                if inter(s[0],u[0],x[0],y[0]) :
                                    bool = False
                                    break
                        if not bool :
                            break
                    if bool :
                        h = round(np.random.random(),1)
                        add_road(N,s,u,h)
    for s in N :
        if s[2] == [] :
            for u in N :
                if u[0] != s[0] :
                    bool = True
                    for x in N :
                        for r in x[2] :
                            y = r[1]
                            if x[4] < y[4] :
                                if inter(s[0],u[0],x[0],y[0]) :
                                    bool = False
                                    break
                        if not bool :
                            break
                    if bool :
                        h = round(np.random.random(),1)
                        add_road(N,s,u,h)
    V = parcours_val(N)
    for i in range(len(V)) :
        for j in V[i] :
            N[j][1] = i
    return N


def parcours_val(N) :
    deja_vu = [False for _ in range(len(N))]
    V = []
    for s in N :
        if not deja_vu[s[4]] :
            L = explore_val(N,s,deja_vu)
            V.append(L)
    return V


def explore_val(N,s,deja_vu) :
    deja_vu[s[4]] = True
    for r in s[2] :
        u = r[1]
        if r[2] == 0 and not deja_vu[u[4]] :
            return [s[4]] + explore_val(N,u,deja_vu)
    return [s[4]]


def draw_net(N) :
    '''Dessine le réseau passé en argument'''
    cmap = get_cmap(nb_valleys(N))
    drawed = [[False for _ in range(len(N))] for _ in range(len(N))]
    for s in N :
        (x1,y1) = s[0]
        for r in s[2] :
            u = r[1]
            if not drawed[s[4]][u[4]] :
                drawed[s[4]][u[4]] = True
                drawed[u[4]][s[4]] = True
                (x2,y2) = u[0]
                if r[3] :
                    plt.plot([x1,x2],[y1,y2],'--',color='black')
                elif r[2] == 0 :
                    plt.plot([x1,x2],[y1,y2],color=cmap(s[1]))
                else :
                    plt.plot([x1,x2],[y1,y2],color='black')
        if s[3] :
            plt.scatter(x1,y1,c = cmap(s[1]),marker='*',s=40)
        else :
            plt.scatter(x1,y1,c = cmap(s[1]),s=40)
    plt.axis('equal')
    

def nb_valleys(N) :
    '''Renvoie le nombre de vallées couvertes par le réseau N'''
    n = 0
    for s in N :
        n = max(n,s[1])
    return n + 1


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)


def test_complexite(f,n,p) :
    '''Teste la complexité d'une fonction f prenant un argument entier sur n cas, p fois chacun'''
    K = []
    T = []
    for k in range(2,n + 1) :
        N = net_alea(k)
        print(k)
        K.append(k)
        t = time()
        for q in range(p) :
            X = f(N)
        T.append((time() - t)/p)
    plt.plot(K,T)
    plt.plot(K,[0.00124/12 * k * k for k in K])
    plt.show()


def cout(N) :
    S = 0
    for s in N :
        for r in s[2] :
            if r[3] :
                S += 6 * r[4]
            else :
                S += r[4]
    return S


def elimination(N) :
    P = deepcopy(N)
    M = net2mat(P)
    Dist = floyd_warshall(M)
    n = len(M)
    Del = [[False for _ in range(n)] for _ in range(n)]
    
    for i in range(n) :
        s = P[i]
        for r in P[i][2] :
            u = r[1]
            j = u[4]
            if r[4] > Dist[i][j] :
                rem_road(P,s,u)
                Del[i][j] = True
    
    return P,Del


def tunnelisation(N) :
    P,Del = elimination(N)
    n = len(P)    
    T = []
    
    for i in range(n) :
        for j in range(i+1,n) :
            M = net2mat(P)
            if M[i][j] == np.infty :
                if not Del[i][j] :
                    Q = deepcopy(P)
                    add_road(Q,Q[i],Q[j],0,True)
                    R,D = elimination(Q)
                    if cout(R) < cout(P) :
                        print(cout(R),cout(P))
                        rem_road(P,P[i],P[j])
                        add_road(P,P[i],P[j],-1,True)
                        P,D = elimination(P)
        
    
    plt.subplot(121)
    draw_net(N)
    
    plt.subplot(122)
    draw_net(P)
    
    plt.show()
    
    return P
    

def super_elimin(N) :
    P = deepcopy(N)
    M = net2mat(P)
    Dist = floyd_warshall(M)
    n = len(M)
    Del = [[False for _ in range(n)] for _ in range(n)]
    
    for i in range(n) :
        s = P[i]
        for r in P[i][2] :
            u = r[1]
            j = u[4]
            if i < j :
                if r[4] > 0.99 * Dist[i][j] :
                    rem_road(P,s,u)
                    Del[i][j] = True
                    M = net2mat(P)
                    Dist = floyd_warshall(M)
    return P


def deep_elimination(N) :
    P = deepcopy(N)
    M = net2mat(P)
    Dist = floyd_warshall(M)
    n = len(M)
    Del = [[False for _ in range(n)] for _ in range(n)]
    
    for i in range(n) :
        s = P[i]
        for r in P[i][2] :
            u = r[1]
            j = u[4]
            if i < j :
                rem_road(P,s,u)
                diff = floyd_warshall(net2mat(P)) - Dist
                S = 0
                for k in range(n) :
                    for l in range(n) :
                        S += diff[k][l]
                        
                if S <= 0 :
                    Del[i][j] = True
                    Del[j][i] = True
                else :
                    add_road(P,s,u,r[2],r[3])
    
    return P,Del


def ex() :
    N = empty_net()
    
    N.append(Node((0,1),0,[],True,0))
    N.append(Node((1,1),1,[],True,1))
    N.append(Node((1,0),2,[],True,2))

    add_road(N,N[0],N[1],1000)
    add_road(N,N[2],N[1],0.5)
    add_road(N,N[0],N[2],0.5)
    
    return N