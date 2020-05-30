import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
plt.style.use('seaborn-whitegrid')


class Node :
    '''Classe définissant un noeud par ses coordonnées, sa vallée et ses routes attachées'''
    
    def __init__(self,coordinates=(0.5,0.5),valley=0,roads=[],town=False,id=-1) :
        self.coord = coordinates
        self.valley = valley
        self.roads = roads
        self.town = town
        self.id = id
    
    def __repr__(self) :
        (x1,y1) = self.coord
        x = round(x1,3)
        y = round(y1,3)
        if self.town :
            aff ="Ville localisée en {}, appartenant à la vallée {}".format((x,y),self.valley)
        else :
            aff = "Croisement localisé en {}, appartenant à la vallée {}".format((x,y),self.valley)
        return aff


class Road :
    '''Classe définissant une route par ses extrémités, sa longueur, la hauteur de son col et son statut de tunnel (ou non)'''
    
    def __init__(self,start,end,height=0,is_tunnel=False) :
        self.start = start
        self.end = end
        self.length = longueur(start,end,height,is_tunnel)
        self.height = height
        self.tunn = is_tunnel
    
    def __repr__(self) :
        (x,y) = self.start.coord
        x1 = round(x,3)
        y1 = round(y,3)
        (x,y) = self.end.coord
        x2 = round(x,3)
        y2 = round(y,3)
        if self.tunn :
            aff ="Tunnel reliant les points {} et {}".format((x1,y1),(x2,y2))
        else :
            aff ="Route reliant les points {} et {}".format((x1,y1),(x2,y2))
            if self.height == 0 :
                aff += " passant par la vallée {}".format(self.start.valley)
            else :
                aff += " passant par un col de hauteur {}".format(round(self.height,3))
        return aff + " et de longueur {}.".format(round(self.length,3))


def longueur(start,end,h,is_tunn=False) :
    '''Calcule approximativement la longueur d'une route allant de start à end et passant par un col de hauteur h'''
    if is_tunn :
        (x1,y1) = start.coord
        (x2,y2) = end.coord
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    elif h == 0 :
        return 0
    else :
        (x1,y1) = start.coord
        (x2,y2) = end.coord
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return np.sqrt(h**2 + L**2) + h * h * np.log((L + np.sqrt(h ** 2 + L ** 2)) / h) / L


def empty_network() :
    '''Crée un réseau routier vide'''
    return []


def add_node(G,coord,valley,is_town,id) :
    '''Ajoute un noeud au réseau G données ses coordonnées, sa vallée et son statut (ville ou non)'''
    s = Node(coord,valley,[],is_town,id)
    G.append(s)


def add_road(G,start,end,height,is_tunnel=False) :
    '''Ajoute une route au réseau donnés son point de départ, son point d'arrivée et la hauteur du col les séparant.'''
    start.roads.append(Road(start,end,height,is_tunnel))
    end.roads.append(Road(end,start,height,is_tunnel))


def rem_road(G,start,end) :
    R = start.roads
    S = end.roads
    
    for r in R :
        if r.end == end :
            R.remove(r)
            break
    
    for s in S :
        if s.end == start :
            S.remove(s)
            break


def net2mat(N) :
    '''Associe à un réseau sa matrice d'adjacence'''
    n = len(N)
    M = np.array([[np.infty for _ in range(n)] for _ in range(n)])
    for i in range(len(N)) :
        for r in N[i].roads :
            j = r.end.id
            M[i][j] = r.length
            M[j][i] = r.length
    for k in range(n) :
        M[k][k] = 0
    return M


def mat2sa(G) :
    n = len(G)
    S = [k for k in range(n)]
    A = []
    for i in range(n) :
        for j in range(i + 1, n) :
            if G[i][j] != np.infty :
                A.append((i,j,G[i][j]))
    return (S,A)


def indice(L,s) :
    '''Renvoie l'indice de l'élément s dans la liste L si il y est et -1 sinon'''
    for k in range(len(L)) :
        if L[k] == s :
            return k
    return -1
            
    
def floyd_warshall(G) :     #Complexité en n^3
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
    N = empty_network()
    P = rd_pt(n)
    add_node(N,P[0],0,False,0)
    for k in range(1,n) :
        s = Node(P[k],k,[],False,k)
        if k == 1 :
            N.append(s)
            add_road(N,N[0],s,np.random.random() * 0.5)
        else :
            N.append(s)
            for u in N :
                if u.coord != s.coord :
                    bool = True
                    for x in N :
                        for r in x.roads :
                            y = r.end
                            if x.id < y.id :
                                if inter(s.coord,u.coord,x.coord,y.coord) :
                                    bool = False
                                    break
                        if not bool :
                            break
                    if bool :
                        h = round(np.random.random(),1)
                        S,U = s.roads,u.roads
                        S.append(Road(s,u,h))
                        U.append(Road(u,s,h))
    V = parcours_val(N)
    for i in range(len(V)) :
        for j in V[i] :
            N[j].valley = i
    return N


def parcours_val(N) :
    deja_vu = [False for _ in range(len(N))]
    V = []
    for s in N :
        if not deja_vu[s.id] :
            L = explore_val(N,s,deja_vu)
            V.append(L)
    return V


def explore_val(N,s,deja_vu) :
    deja_vu[s.id] = True
    for r in s.roads :
        u = r.end
        if r.height == 0 and not deja_vu[u.id] :
            return [s.id] + explore_val(N,u,deja_vu)
    return [s.id]


def draw_net(N) :
    '''Dessine le réseau passé en argument'''
    cmap = get_cmap(nb_valleys(N))
    for s in N :
        (x1,y1) = s.coord
        for r in s.roads :
            if r.end.id > s.id :
                u = r.end
                (x2,y2) = u.coord
                if r.tunn :
                        plt.plot([x1,x2],[y1,y2],'--',color='black')
                elif r.height == 0 :
                    plt.plot([x1,x2],[y1,y2],color=cmap(s.valley))
                else :
                    plt.plot([x1,x2],[y1,y2],color='black')
        if s.town :
            plt.scatter(x1,y1,c = cmap(s.valley),marker='*',s=40)
        else :
            plt.scatter(x1,y1,c = cmap(s.valley),s=40)
    plt.axis('equal')
    

def nb_valleys(N) :
    '''Renvoie le nombre de vallées couvertes par le réseau N'''
    n = 0
    for s in N :
        n = max(n,s.valley)
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
        M = net2mat(N)
        print(k)
        K.append(k)
        t = time()
        for q in range(p) :
            X = f(M)
        T.append((time() - t)/p)
    plt.plot(K,T)
    plt.plot(K,[0.0000025 * k** 3 for k in K])
    plt.show()


def cout_construction(N) :  #A nln(n) avec A = 6.25
    S = 0
    for s in N :
        for r in s.roads :
            if r.tunn :
                S += 2 * r.length
            else :
                S += r.length
    return S


def cout_usage(N) :     #  B n²
    Dist = floyd_warshall(net2mat(N))
    S = 0
    for i in range(len(N)) :
        for j in range(i,len(N)) :
            S += Dist[i][j]
    return S / len(N) ** 0.5


def delta(N,P) :
    D = floyd_warshall(net2mat(N))
    E = floyd_warshall(net2mat(P))
    M = abs(E[0][0] - D[0][0])
    n = len(N)
    
    for i in range(n) :
        for j in range(i+1,n) :
            m = abs(E[i][j] - D[i][j])
            M = max(m,M)
    
    return M


def elimination(N) :    #Complexité amortie entre x^2 et x^3
    P = deepcopy(N)
    M = net2mat(P)
    Dist = floyd_warshall(M)
    n = len(M)
    Del = [[False for _ in range(n)] for _ in range(n)]
    
    for i in range(n) :
        s = P[i]
        for r in P[i].roads :
            u = r.end
            j = u.id
            if i < j :
                if r.length > Dist[i][j] :
                    rem_road(P,s,u)
                    Del[i][j] = True
    
    return P,Del


def tunnelisation(N) :  #Complexité amortie : presque en n^4
    P,Del = elimination(N)
    n = len(P)   
    p = cout_construction(P)
    
    for i in range(n) :
        for j in range(i+1,n) :
            M = net2mat(P)
            if M[i][j] == np.infty and i < j :
                if not Del[i][j] :
                    Q = deepcopy(P)
                    add_road(Q,Q[i],Q[j],-1,True)
                    R,D = elimination(Q)
                    c_r = cout_construction(R)
                    if c_r < p :
                        p = c_r
                        add_road(P,P[i],P[j],-1,True)
                        P,Del = elimination(P)
    
    return P


def test_integral(N) :
    
    plt.subplot(221)
    draw_net(N)
    cc,cu = cout_construction(N),cout_usage(N)
    print(cc,cu)
    plt.grid(False)
    
    O = elimination(N)[0]
    plt.subplot(222)
    draw_net(O)
    print(cout_construction(O),cout_usage(O))
    plt.grid(False)
    
    i = compare_couts(O,False)
    if len(i) == 1 :
        Q = O
    else :
        Q = elimination(detriangularisation(O,i[1]))[0]
    i = compare_couts(Q,False)
    if len(i) == 1 :
        S = Q
    else :
        S = elimination(detriangularisation(Q,i[1]))[0]
    plt.subplot(223)
    draw_net(S)
    print(cout_construction(S),cout_usage(S))
    plt.grid(False)    
    
    R = elimination(tunnelisation(S))[0]
    i = compare_couts(R,False)
    if len(i) == 1 :
        P = S
    else :
        P = elimination(detriangularisation(R,i[1]))[0]
    plt.subplot(224)
    draw_net(P)
    cc1,cu1 = cout_construction(P),cout_usage(P)
    print(cc1,cu1)
    plt.grid(False)
    
    print((cc-cc1)/cc,(cu1-cu)/cu)
    plt.show()
    return O,P,Q


def critere(crit,au,ac,bu,bc,cu,cc) :
    if ac + bc >= (1 + crit) * cc and au + bu <= (2 - crit) * cu :
        return True
    return False


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
    for r in N[i].roads :
        
        if r.end.id == j :
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
    
    return N


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


def compare_couts(N,bool=True) :
    n = len(N)
    P = elimination(N)[0]
    K = np.linspace(0,1,50)
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


def test(n,p) :
    K = []
    T = []
    for k in range(2,n + 1) :
        print(k)
        K.append(k)
        S = 0
        i= 0
        for q in range(p) :
            N = net_alea(k)
            Q,d = elimination(N)
            c = cout_construction(Q)
            if c != np.inf :
                i += 1
                S += c
        T.append(S/(k* i))
    plt.plot(K,T)
    plt.show()


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
        

def mat_construction(N) :
    n = len(N)
    M = np.array([[np.infty for _ in range(n)] for _ in range(n)])
    for i in range(len(N)) :
        for r in N[i].roads :
            j = r.end.id
            if r.tunn :
                M[i][j] = 2 * r.length
                M[j][i] = 2 * r.length
            else :
                M[i][j] = r.length
                M[j][i] = r.length
    for k in range(n) :
        M[k][k] = 0
    return M


