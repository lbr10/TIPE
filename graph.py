def graph() :
    return []

def is_empty(graph) :
    return graph == []

def add_vert(graph,list) :
    n = len(graph)
    graph.append(list)
    for k in list :
        graph[k].append(n)
    return graph

def add_edge(graph,i,j) :
    graph[i].append(j)
    graph[j].append(i)
    return graph

def adj(graph,i) :
    return graph[i]

