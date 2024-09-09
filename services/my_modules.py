import random
from heapq import heappush, heappop

# リストからタプルへ変換 O(NlogN)
def list_to_tuple(x):
    return tuple(list_to_tuple(item) if isinstance(item, list) else item for item in x)

# タプルからリストへ変換 O(NlogN)
def tuple_to_list(x):
    return list(tuple_to_list(item) if isinstance(item, tuple) else item for item in x)

# グラフをランダムに作成
def create_random_arcs(n, m, s, t, budget):
    X = [i for i in range(n-1)]
    Y = [n-1]
    G = [[] for i in range(n)]
    G_node = [[] for i in range(n)]
    res_Arcs = [0 for i in range(n)]
    not_node = []

    for i in range(n-1):
        y = random.choice(Y)
        x = random.choice(X)
        w = random.randint(1, 10)
        p = random.randint(1, 10)
        G[x].append([y, w, p])
        G_node[x].append(y)
        Y.append(x)
        X.remove(x)
        res_Arcs[x] += 1
        if res_Arcs[x] == n-1:
            not_node.append(x)

    for i in range(m - n + 1):
        for j in not_node:
            Y.remove(j)
        Y.remove(t)
        a = random.choice(Y)
        for j in not_node:
            Y.append(j)
        Y.append(t)
        Y.remove(a)
        for j in G_node[a]:
            Y.remove(j)
        b = random.choice(Y)
        Y.append(a)
        for j in G_node[a]:
            Y.append(j)
        # if not(a == s and b == t) or n == 2:
        w = random.randint(1, 10)
        p = random.randint(1, 10)
        G[a].append([b, w, p])
        G_node[a].append(b)
        res_Arcs[a] += 1
        if res_Arcs[a] == n-1:
            not_node.append(a)

    # for i in range(n):
        # G[i].sort()

    input_list = [n, m, s, t, budget]
    for i in range(n):
        for j in range(len(G[i])):
            input_list.append(i)
            input_list.append(G[i][j][0])
            input_list.append(G[i][j][1])
            input_list.append(G[i][j][2])
    return input_list

# ダイクストラ法 O(N+MlogN)
def dijkstra(s, n, INF, G_rev):
    dist = [INF] * n
    prev = [-1] * n
    hq = [(0, s)]
    dist[s] = 0
    visited = [False] * n
    while hq:
        i = heappop(hq)[1]
        if visited[i]:
            continue
        visited[i] = True
        for j, k in G_rev[i]:
            if not visited[j] and dist[i] + k < dist[j]:
                dist[j] = dist[i] + k
                prev[j] = i
                heappush(hq, (dist[j], j))
    return prev, dist

# all-to-t-shortest path tree Tの作成 O(N**2)
def get_T(n, prev):
    T = set()
    for i in range(n - 1):
        cur = i
        while cur != -1 and cur != n - 1:
            T.add((cur, prev[cur]))
            cur = prev[cur]
    return T
