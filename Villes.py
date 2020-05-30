import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
plt.style.use('seaborn-whitegrid')
from mpl_toolkits import mplot3d


def Town(coords,val,rd,ps,i) :
    s = {}
    s['coord'] = coords
    s['valley'] = val
    s['roads'] = rd
    s['passes'] = ps
    s['id'] = i
    return s


def Road(strt,en,tunn) :
    r = {}
    r['start'] = strt
    r['end'] = en
    r['is_tunn'] = tunn
    (x1,y1,z1) = strt['coord']
    (x2,y2,z2) = en['coord']
    r['length'] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    r['pas'] = False
    return r


def Pas(start,end,top) :
    p = {}
    p['start'] = start
    p['end'] = end
    p['top'] = top
    p['length'] = longueur(start,top) + longueur(top,end)
    p['pas'] = True
    return p


def longueur(start,end) :
    (x1,y1,z1) = start['coord']
    (x2,y2,z2) = end['coord']
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if np.arctan(abs(z2 - z1) / L) > np.pi / 9 :
        return abs(z2 - z1) / np.sin(np.pi / 9)
    else :
        return np.sqrt(L ** 2 + (z2 - z1) ** 2)


def add_road(start,end,is_tunn) :
    start['roads'].append(Road(start,end,is_tunn))
    end['roads'].append(Road(end,start,is_tunn))


def add_pas(start,end,top) :
    start['passes'].append(Pas(start,end,top))
    end['passes'].append(Pas(end,start,top))


def net2mat(N) :
    '''Associe à un réseau sa matrice d'adjacence'''
    n = len(N)
    M = [[np.infty for _ in range(n)] for _ in range(n)]
    
    for i in range(len(N)) :
        for r in N[i]['roads'] :
            j = r['end']['id']
            M[i][j] = r['length']
            M[j][i] = r['length']
        for r in N[i]['passes'] :
            j = r['end']['id']
            M[i][j] = r['length']
            M[j][i] = r['length']
    for k in range(n) :
        M[k][k] = 0
    return M


def floyd_warshall(G) :     #Complexité en n^3
    '''Renvoie le distancier d'un graphe défini par sa matrice d'adjacence'''
    W = deepcopy(G)
    n = len(W)
    for k in range(n) :
        for i in range(n) :
            for j in range(n) :
                W[i][j] = min(W[i][j],W[i][k] + W[k][j])
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
        P.append((5 * (np.random.random() - 0.5 + x), 5 * (np.random.random() - 0.5 + y), 1.5 * np.random.random()))
    
    return P


def inter(a,b,c,d) :
    (xa,ya,_) = a
    (xb,yb,_) = b
    (xc,yc,_) = c
    (xd,yd,_) = d
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


def top_alea(start,end) :
    (x1,y1,z1) = start['coord']
    (x2,y2,z2) = end['coord']
    x = min(x1,x2) + np.random.random() * (max(x1,x2) - min(x1,x2))
    y = min(y1,y2) + np.random.random() * (max(y1,y2) - min(y1,y2))
    z = round(max(z1,z2) + np.random.random() / 2)
    top = Town((x,y,z),-1,[],[],-1)

    return top



def net_alea(n,p) :
    '''Crée un réseau aléatoire avec n noeuds'''
    N = []
    P = rd_pt(n)
    N.append(Town(P[0],0,[],[],0))
    
    for k in range(1,n) :
        
        if np.random.random() < p :
            s = Town(P[k],k,[],[],k)
        else :
            s = Town(P[k],k,[],[],k - n)
        
        if k == 1 :
            N.append(s)
            
            if np.random.random() > 0.95 :
                s['valley'] = 0
                add_road(N[0],s,False)
            else :
                add_pas(N[0],s,top_alea(N[0],s))
            
        else :
            N.append(s)
            for u in N :
                if u['coord'] != s['coord'] :
                    bool = True
                    for x in N :
                        for r in (x['roads'] + x['passes']) :
                            y = r['end']
                            if abs(x['id']) < abs(y['id']) :
                                if inter(s['coord'],u['coord'],x['coord'],y['coord']) :
                                    bool = False
                                    break
                        if not bool :
                            break
                    if bool :
                        if np.random.random() > 0.95 :
                            s['valley'] = u['valley']
                            add_road(u,s,False)
                        else :
                            add_pas(u,s,top_alea(u,s))
    return N


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)


def draw_net(N) :
    '''Dessine le réseau passé en argument'''
    n = len(N)
    cmap = get_cmap(n)
    for s in N :
        (x1,y1,z1) = s['coord']
        for r in s['roads'] :
            if abs(r['end']['id']) > abs(s['id']) :
                u = r['end']
                (x2,y2,z2) = u['coord']
                if r['is_tunn'] :
                    plt.plot([x1,x2],[y1,y2],linestyle='--',color='black')
                else :
                    plt.plot([x1,x2],[y1,y2],color=cmap(s['valley']))
                    
        for r in s['passes'] :
            if abs(r['end']['id']) > abs(s['id']) :
                u = r['end']
                (x2,y2,z2) = u['coord']
                plt.plot([x1,x2],[y1,y2],color='black')
                
        if s['id'] >= 0 :
            plt.scatter(x1,y1,c = cmap(s['valley']),s=40)
    plt.axis('equal')


def distancier(N) :
    G = net2mat(N)
    Dist = floyd_warshall(G)
    n = len(N)
    V = []
    
    for i in range(n) :
        if N[i]['id'] >= 0 :
            V.append(i)
    
    v = len(V)
    Villes = np.zeros((v,v))
    
    for i in range(v) :
        for j in range(v) :
            Villes[i][j] = Dist[V[i]][V[j]]
    
    return Villes


def cout_construction(N) :
    S = 0
    for s in N :
        for r in s['roads'] :
            if r['is_tunn'] :
                S += 2 * r['length']
            else :
                S += r['length']
        for p in s['passes'] :
            S += p['length']
    return S / 2


def cout_usage(N) :
    Dist = distancier(N)
    S = 0
    for i in range(len(Dist)) :
        for j in range(i,len(Dist)) :
            S += Dist[i][j]
    return S / (len(Dist) ** 0.5)


# Elimination de croisements
# Elimination d'arêtes superficielle (vallées)
# Elimination d'arêtes profonde ??? (probabiliste ? méthodique ?)
# Tunnelisation
# Détriangularisation


def mat_construction(N) :
    n = len(N)
    M = np.array([[np.infty for _ in range(n)] for _ in range(n)])
    for i in range(n) :
        for r in N[i]['roads'] :
            j = r['end']['id']
            if r['is_tunn'] :
                M[i][j] = 2 * r['length']
                M[j][i] = 2 * r['length']
            else :
                M[i][j] = r['length']
                M[j][i] = r['length']
        for r in N[i]['passes'] :
            j = r['end']['id']
            M[i][j] = r['length']
            M[j][i] = r['length']
    for k in range(n) :
        M[k][k] = 0
    return M
    

def est_connexe(N) :
    n = len(N)
    deja_vu = [False for _ in range(n)]
    explore(N[0],N,deja_vu)
    return deja_vu == [True for _ in range(n)]


def explore(s,N,deja_vu) :
    i = s['id']
    deja_vu[i] = True
    for r in s['roads'] :
        j = r['end']['id']
        if not deja_vu[j] :
            explore(r['end'],N,deja_vu)
    for r in s['passes'] :
        j = r['end']['id']
        if not deja_vu[j] :
            explore(r['end'],N,deja_vu)


def elimination(N) :
    P = deepcopy(N)
    n = len(P)
    Q = deepcopy(P)
    c = cout_usage(P)
    
    for i in range(n) :
        
        s = Q[i]
        
        for r in s['roads'] :
            u = r['end']
            j = u['id']
            u['roads'].remove(Road(u,s,r['is_tunn']))
            s['roads'].remove(r)
            if cout_usage(Q) <= c :
                P = Q
                c = cout_usage(P)
            else :
                u['roads'].append(Road(u,s,r['is_tunn']))
                s['roads'].append(r)
        
        for r in s['passes'] :
            u = r['end']
            j = u['id']
            u['passes'].remove(Pas(u,s,r['top']))
            s['passes'].remove(r)
            if cout_usage(Q) <= c :
                P = Q
                c = cout_usage(P)
            else :
                u['passes'].append(Pas(u,s,r['top']))
                s['passes'].append(r)
    
    tailladeur(P)
    
    return P


def tailladeur(N) :
    n = len(N)
    i = 0
    while i < n - 1 :
        i += 1
        s = N[i]
        if s['id'] < 0 and len(s['roads']) + len(s['passes']) == 1 :
            for r in s['roads'] :
                u = r['end']
                j = u['id']
                u['roads'].remove(Road(u,s,r['is_tunn']))
            s['roads'] = []
            for r in s['passes'] :
                u = r['end']
                j = u['id']
                u['passes'].remove(Pas(u,s,r['top']))
            s['passes'] = []
            i = 0


def tunnelisation(N) :
    P = deepcopy(N)
    n = len(P)   
    p = cout_construction(P)
    M = net2mat(P)
    Q = deepcopy(P)
        
    for i in range(n) :
        s = Q[i]
        if len(s['passes'] + s['roads']) != 0 :
            for j in range(i+1,n) :
                u = Q[j]
                if len(u['passes'] + u['roads']) != 0 :
                    if M[i][j] == np.infty :
                        u['roads'].append(Road(u,s,True))
                        s['roads'].append(Road(s,u,True))
                        R = elimination(Q)
                        r = cout_construction(R)
                        if r <= p :
                            P = R
                            Q = R
                            p = r
                        else :
                            u['roads'].remove(Road(u,s,True))
                            s['roads'].remove(Road(s,u,True))
    tailladeur(P)
    return P


def cherche_triangles(N) :
    n = len(N)
    T = []
    for i in range(n) :
        for j in range(i+1,n) :
            for k in range(j+1,n) :
                if is_triangle(N,i,j,k) :
                    T.append((i,j,k))
    return T


def is_triangle(N,i,j,k) :
    if is_segment(N,i,j) :
        if is_segment(N,i,k) :
            if is_segment(N,j,k) :
                return True
    return False


def is_segment(N,i,j) :
    for r in N[i]['roads'] :
        if r['end']['id'] == j :
            return True
    for r in N[i]['passes'] :
        if r['end']['id'] == j :
            return True
    return False


def nettoie_triangles(T,j,k) :
    U = []
    for t in T :
        (a,b,c) = t
        if a != j :
            if b != k and c != k :
                U.append(t)
        elif b != j :
            if c != k :
                U.append(t)
    return U


def critere(crit,au,ac,bu,bc,cu,cc) :
    if ac + bc >= (1 + crit) * cc and au + bu <= (2 - crit) * cu :
        return True
    return False


def detriangularisation(P,crit) :    #Complexité amortie : quadratique
    N = deepcopy(P)
    T = cherche_triangles(N)
    C = mat_construction(N)
    
    while T != [] :
        M = net2mat(N)
        (i,j,k) = T.pop()
        if critere(crit,M[i][j],C[i][j],M[i][k],C[i][k],M[j][k],C[j][k]) :
            rem_road(N,N[j],N[k])
            T = nettoie_triangles(T,j,k)
        elif critere(crit,M[i][j],C[i][j],M[j][k],C[j][k],M[i][k],C[i][k]) :
            rem_road(N,N[i],N[k])
            T = nettoie_triangles(T,i,k)
        elif critere(crit,M[j][k],C[j][k],M[i][k],C[i][k],M[i][j],C[i][j]) :
            rem_road(N,N[i],N[j])
            T = nettoie_triangles(T,i,j)
    tailladeur(N)
    return N


def rem_road(N,s,u) :
    for r in s['roads'] :
        if r['end'] == u :
            s['roads'].remove(r)
            u['roads'].remove(Road(u,s,r['is_tunn']))
    for r in s['passes'] :
        if r['end'] == u :
            s['passes'].remove(r)
            u['passes'].remove(Pas(u,s,r['top']))


def recherche_min(X,Y) :
    m = Y[0]
    L = [0]
    
    for i in range(len(X)) :
        if m > Y[i] :
            m = Y[i]
            L = [X[i]]
        elif m == Y[i] :
            L.append(X[i])
            
    
    return L


def compare_couts(N,bool=True) :
    n = len(N)
    P = elimination(N)
    K = np.linspace(0,1,10)
    L = []
    M = []
    R = []
    O = []
    Q = []
    A = []
    cun = cout_usage(P)
    ccn = cout_construction(P)
    for k in K :
        cc = cout_construction(detriangularisation(P,k))
        L.append(cc)
        Q.append((100 * cc/ccn) - 100)
        cu = (cout_usage(detriangularisation(P,k)))
        M.append(cu)
        O.append((100 * cu/cun) - 100)
        A.append(100 * abs((cu-cun)**0.5/(cc-ccn)))
    if bool :
        plt.plot(K,Q,label='cc')
        plt.plot(K,O,label='cu')
        plt.plot(K,A)
        plt.legend()
        plt.show()
    return recherche_min(K,A)


def test_integral(N) :
    
    plt.subplot(221)
    draw_net(N)
    cc,cu = cout_construction(N),cout_usage(N)
    print(int(cc),int(cu))
    plt.grid(False)
    
    O = elimination(N)
    plt.subplot(222)
    draw_net(O)
    print(int(cout_construction(O)),int(cout_usage(O)))
    plt.grid(False)
    
    i = compare_couts(O,False)
    if len(i) == 1 :
        Q = O
    else :
        Q = elimination(detriangularisation(O,i[1]))
    i = compare_couts(Q,False)
    if len(i) == 1 :
        S = Q
    else :
        S = elimination(detriangularisation(Q,i[1]))
    plt.subplot(223)
    draw_net(S)
    print(int(cout_construction(S)),int(cout_usage(S)))
    plt.grid(False)    
    
    R = elimination(tunnelisation(S))
    i = compare_couts(R,False)
    if len(i) == 1 :
        P = S
    else :
        P = elimination(detriangularisation(R,i[1]))
    plt.subplot(224)
    draw_net(P)
    cc1,cu1 = cout_construction(P),cout_usage(P)
    print(int(cc1),int(cu1))
    plt.grid(False)
    
    print(round((cc-cc1)/cc,3),round((cu1-cu)/cu,3))
    plt.show()
    return O,P,Q



