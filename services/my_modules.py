from heapq import heappush, heappop

# リストからタプルへ変換 O(NlogN)
def list_to_tuple(x):
    return tuple(list_to_tuple(item) if isinstance(item, list) else item for item in x)

# タプルからリストへ変換 O(NlogN)
def tuple_to_list(x):
    return list(tuple_to_list(item) if isinstance(item, tuple) else item for item in x)

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
