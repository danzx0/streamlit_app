import matplotlib.pyplot as plt
from heapq import heappush, heappop
from collections import defaultdict
import itertools
import copy

import networkx as nx

# import my_networkx as my_nx

# リストからタプルへ変換 O(NlogN)
def list_to_tuple(x):
    return tuple(list_to_tuple(item) if isinstance(item, list) else item for item in x)

# タプルからリストへ変換 O(NlogN)
def tuple_to_list(x):
    return list(tuple_to_list(item) if isinstance(item, tuple) else item for item in x)

# ダイクストラ法 O(N+MlogN)
def dijkstra(s, n):
    INF = 10 ** 9
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
        for j, k in G_reverse[i]:
            if not visited[j] and dist[i] + k < dist[j]:
                dist[j] = dist[i] + k
                prev[j] = i
                heappush(hq, (dist[j], j))
    return prev, dist

# all-to-t-shortest path tree Tの作成 O(N**2)
def get_T(prev):
    T = set()
    for i in range(n-1):
        cur = i
        while cur != -1 and cur != n-1:
            T.add((cur, prev[cur]))
            cur = prev[cur]
    return T

# 入力する必要があるもの
# ノード数, アーク数, 出発点, 到着点, 阻止の予算
# n, m, s, t, b = , , , ,
# グラフ(隣接リスト)[始点, 終点, コスト, 増加分]
# G = [[[, , , ], [, , , ]], [[, , , ], [, , , ]], ..., []]
# 逆向きのグラフ(隣接リスト)(終点, コスト)
# G_reverse = [[], [(, ), (, )], ..., [(, )]]
# アークの集合(始点, 終点, コスト, 増加分)
# A = ((, , , ), (, , , ), ..., (, , , ))
# アークの集合(始点, 終点)
# Arcs = ((, ), (, ), ..., (, ))

# データの代入例1
# 5 9 0 4 2 0 1 2 2 0 2 6 2 1 2 3 4 1 3 1 2 1 4 4 8 2 1 4 2 2 4 2 4 3 1 2 4 3 4 4 4
# ノード数, アーク数, 出発点, 到着点, 阻止の予算
# n, m, s, t, b = 5, 9, 0, 4, 2
# # グラフ(隣接リスト)[始点, 終点, コスト, 増加分]
# G = [[[0, 1, 2, 2], [0, 2, 6, 2]], [[1, 2, 3, 4], [1, 3, 1, 2], [1, 4, 4, 8]], [[2, 1, 4, 2], [2, 4, 2, 4]], [[3, 1, 2, 4], [3, 4, 4, 4]], []]
# # 逆向きのグラフ(隣接リスト)(終点, コスト)
# G_reverse = [[], [(0, 2), (2, 4), (3, 2)], [(0, 6), (1, 3)], [(1, 1)], [(1, 4), (2, 2), (3, 4)]]
# # アークの集合(始点, 終点, コスト, 増加分)
# A = ((0, 1, 2, 2), (0, 2, 6, 2), (1, 2, 3, 4), (1, 3, 1, 2), (1, 4, 4, 8), (2, 1, 4, 2), (2, 4, 2, 4), (3, 1, 2, 4), (3, 4, 4, 4))
# # アークの集合(始点, 終点)
# Arcs = ((0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4))

# データの代入例2
# 5 6 0 4 2 0 1 1 2 0 2 1 2 1 4 1 10 2 3 1 2 2 4 3 1 3 4 1 10
# # ノード数, アーク数, 出発点, 到着点, 阻止の予算
# n, m, s, t, b = 5, 6, 0, 4, 2
# # グラフ(隣接リスト)[始点, 終点, コスト, 増加分]
# G = [[[0, 1, 1, 2], [0, 2, 1, 2]], [[1, 4, 1, 10]], [[2, 3, 1, 2], [2, 4, 3, 1]], [[3, 4, 1, 10]], []]
# # 逆向きのグラフ(隣接リスト)(終点, コスト)
# G_reverse = [[], [(0, 1)], [(0, 1)], [(2, 1)], [(1, 1), (2, 3), (3, 1)]]
# # アークの集合(始点, 終点, コスト, 増加分)
# A = ((0, 1, 1, 2), (0, 2, 1, 2), (1, 4, 1, 10), (2, 3, 1, 2), (2, 4, 3, 1), (3, 4, 1, 10))
# # アークの集合(始点, 終点)
# Arcs = ((0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 4))
def get_DSPI_free(input_list):
    # input_list = list(map(int, input().split()))
    n, m, s, t, budget = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
    # graph = nx.DiGraph()
    # edge_list = []
    G = [[] for i in range(n)]
    G_reverse = [[] for i in range(n)]
    # A, Arcs = (), ()
    list_A, list_Arcs = [], []

    for i in range(m):
        a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
        # edge_list.append((a, b, {'weight':w, 'plus':p}))
        G[a].append([a, b, w, p])
        G_reverse[b].append([a, w])
        # A += ((a, b, w, p), )
        # Arcs += ((a, b), )
        list_A.append((a, b, w, p))
        list_Arcs.append((a, b))

    for i in range(n):
        G[i].sort()
        G_reverse[i].sort()
    list_A.sort()
    list_Arcs.sort()
    tuple_A = list_to_tuple(list_A)
    tuple_Arcs = list_to_tuple(list_Arcs)

    # print('n, m, s, t, budget')
    # print(n, m, s, t, budget)
    # print('G')
    # print(G)
    # print('G_reverse')
    # print(G_reverse)
    # print('tuple_A')
    # print(tuple_A)
    # print('tuple_Arcs')
    # print(tuple_Arcs)

    # z_star, z_barの定義 (キー : tuple, 要素 : list)
    z_star = defaultdict(list)
    z_bar = defaultdict(list)
    next_z_star = defaultdict(list)
    next_z_bar = defaultdict(list)

    # 何も阻止されていない場合で逆向きのダイクストラを行い, 前にいた頂点のリストとコストを出力
    prev, d = dijkstra(n-1, n)
    # print('Sが空集合のときのprev')
    # print(prev)
    # print('Sが空集合のときのtからのコスト')
    # print(d)

    # 経路を復元して, all-to-t-shortest path tree Tを作成
    T = get_T(prev)
    # print('T')
    # print(T)

    # |S|==bを満たすSの集合cを定義する
    c = list(itertools.combinations(tuple_A, budget))
    # print('c')
    # print(c)

    # |S|==bの各Sについて, Tの辺を含むか判定し, 含む場合はGを定義してダイクストラを行う.
    for S in c:
        G_reverse = [[] for i in range(n)]
        flag = False
        for j in tuple_A:
            if j in S:
                if (j[0], j[1]) in T: # Tに含まれているかどうかの判定
                    flag = True
                G_reverse[j[1]].append((j[0], j[2] + j[3]))
            else:
                G_reverse[j[1]].append((j[0], j[2]))
        if flag:
            prev, list_z_star_S = dijkstra(n-1, n)
            # print('S, z_star_S, prev')
            # print(S, list_z_star_S, prev)
            z_star[S] = list_z_star_S
            next_z_star[S] = [[(), -1] for i in range(n)]
            # print(next_z_star[S])
            for i in range(n-1):
                next_z_star[S][i] = (S, prev[i])
    # print('z_star')
    # print(z_star)

    # DP-DSPIの2行目
    for k in range(budget-1, -1, -1):
        # DP-DSPIの3行目
        c = list(itertools.combinations(tuple_A, k))
        for S in c:
            z_bar[S] = [0] * n # continueしたものは0が代入される
            next_z_bar[S] = [[(), -1] for i in range(n)]
            next_z_star[S] = [[(), -1] for i in range(n)]
            list_S = tuple_to_list(S)
            # print('list_S')
            # print(list_S)
            for i in range(n):
                A_i = copy.deepcopy(G[i])
                # print('A_i')
                # print(A_i)
                # A_iの要素のうち, すでにSに含まれているものを削除
                for item in list_S:
                    if item in A_i:
                        A_i.remove(item)
                num = len(A_i)
                # print('list_S, i, A_i')
                # print(list_S, i, A_i)
                if num > 0:
                    # bit全探索でS_primeを全列挙する
                    max_val = 0
                    new_max_S = tuple()
                    new_max_i = -1
                    for j in range(1, 2**num):
                        new_S = copy.deepcopy(list_S)
                        for l in range(num):
                            if j >> l & 1:
                                # S_primeに追加する
                                new_S.append(A_i[l])
                        # 予算を超えていたら次のnew_Sを探索
                        if  len(new_S) > budget:
                            continue
                        new_S.sort()
                        new_S = list_to_tuple(new_S)
                        # z_star[new_S]が存在するかの判定
                        if z_star[new_S] == []:
                            # print('含まれないの発見')
                            # print(S, i, new_S)
                            continue
                        min_val = INF
                        # new_min_S = new_S
                        for list_arc in G[i]:
                            # c_tildeの更新
                            arc = list_to_tuple(list_arc)
                            if arc in new_S:
                                c_tilde = arc[2] + arc[3]
                            else:
                                c_tilde = arc[2]
                            if min_val > z_star[new_S][arc[1]] + c_tilde:
                                min_val = z_star[new_S][arc[1]] + c_tilde
                                new_min_i = arc[1]
                        if max_val < min_val:
                            max_val = min_val
                            new_max_S = new_S
                            new_max_i = new_min_i
                    z_bar[S][i] = max_val
                    next_z_bar[S][i] = (new_max_S, new_max_i)
            # print('z_bar[' + str(S) + ']')
            # print(z_bar[S])
            l = [INF] * n
            l[t] = 0
            l_caron = [INF] * n
            l_caron[t] = 0
            l_Q = [(0, t)]
            for i in range(t):
                heappush(l_Q, (INF, i))
            # すでに訪れたかどうかを保持しておく
            visited = [False] * n
            while l_Q:
                j = heappop(l_Q)
                if visited[j[1]]:
                    continue
                visited[j[1]] = True
                for i in range(n):
                    if (i, j[1]) in tuple_Arcs:
                        res = tuple_Arcs.index((i, j[1]))
                        if tuple_A[res] in S:
                            c_tilde = tuple_A[res][2] + tuple_A[res][3]
                        else:
                            c_tilde = tuple_A[res][2]
                        if l_caron[i] > j[0] + c_tilde and l_caron[i] > z_bar[S][i]:
                            l_caron[i] = j[0] + c_tilde
                            if l_caron[i] < z_bar[S][i]:# 阻止する場合
                                l[i] = z_bar[S][i]
                                next_z_star[S][i] = (next_z_bar[S][i][0], next_z_bar[S][i][1])
                            else:# 阻止しない場合
                                l[i] = l_caron[i]
                                next_z_star[S][i] = [S, j[1]]
                            heappush(l_Q, (l[i], i))
            ans = copy.deepcopy(l)
            z_star[S] = ans
            # print('z_star[S]')
            # print(z_star[S])
    # print('z_bar')
    # print(z_bar)
    # for i in z_bar.items():
    #     print(i)
    # print('z_star')
    # print(z_star)
    # for i in z_star.items():
    #     print(i)
    return z_star, next_z_star, tuple_Arcs, tuple_A

# now_node = 0
# now_S = tuple()
# now_cost = 0
# # グラフの描画準備
# graph = nx.DiGraph()
# for i in range(m):
#     a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
#     edge_list.append((a, b, {'weight':w, 'plus':p}))

# graph.add_edges_from(edge_list)
# pos  = nx.kamada_kawai_layout(graph)
# for i in range(n):
#     pos[i][0] = pos[i][0] * (-1)
# fig, ax = plt.subplots()
# nx.draw_networkx_nodes(graph, pos, ax = ax)
# nx.draw_networkx_labels(graph, pos, ax = ax)

# curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
# straight_edges = list(set(graph.edges()) - set(curved_edges))
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
# arc_rad = 0.29
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

# edge_weights = nx.get_edge_attributes(graph,'weight')
# edge_plus = nx.get_edge_attributes(graph,'plus')
# curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
# straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
# my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad = arc_rad)
# nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
# plt.show()
# interdicted_curved_edges = []
# interdicted_straight_edges = []
# passed_curved_edges = []
# passed_interdicted_curved_edges = []
# passed_straight_edges = []
# passed_interdicted_straight_edges = []
# while now_node != t:
#     next_S_and_node = next_z_star[now_S][now_node]
#     next_S = next_S_and_node[0]
#     next_node = next_S_and_node[1]
#     now_cost += z_star[now_S][now_node] - z_star[next_S][next_node]
#     print('阻止集合')
#     print(next_S)
#     print('今回の移動')
#     print(str(now_node) + '→' + str(next_node))
#     print('ここまでの総コスト')
#     print(now_cost)
#     # グラフの描画
#     fig, ax = plt.subplots()
#     nx.draw_networkx_nodes(graph, pos, ax = ax)
#     nx.draw_networkx_labels(graph, pos, ax = ax)

#     # curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
#     # straight_edges = list(set(graph.edges()) - set(curved_edges))

#     # interdicted_curved_edges = []
#     for i in range(len(curved_edges)):
#         res = tuple_Arcs.index(curved_edges[i])
#         if tuple_A[res] in next_S:
#             interdicted_curved_edges.append(curved_edges[i])
#     for i in interdicted_curved_edges:
#         if i in curved_edges:
#             curved_edges.remove(i)

#     # interdicted_straight_edges = []
#     for i in range(len(straight_edges)):
#         res = tuple_Arcs.index(straight_edges[i])
#         if tuple_A[res] in next_S:
#             interdicted_straight_edges.append(straight_edges[i])
#     for i in interdicted_straight_edges:
#         if i in straight_edges:
#             straight_edges.remove(i)

#     pass_curved_edge = []
#     pass_interdicted_curved_edge = []
#     pass_straight_edge = []
#     pass_interdicted_straight_edge = []
#     if (now_node, next_node) in curved_edges:
#         pass_curved_edge.append((now_node, next_node))
#         curved_edges.remove((now_node, next_node))
#         passed_curved_edges.append((now_node, next_node))
#     elif (now_node, next_node) in interdicted_curved_edges:
#         pass_interdicted_curved_edge.append((now_node, next_node))
#         interdicted_curved_edges.remove((now_node, next_node))
#         passed_interdicted_curved_edges.append((now_node, next_node))
#     elif (now_node, next_node) in straight_edges:
#         pass_straight_edge.append((now_node, next_node))
#         straight_edges.remove((now_node, next_node))
#         passed_straight_edges.append((now_node, next_node))
#     else:
#         pass_interdicted_straight_edge.append((now_node, next_node))
#         interdicted_straight_edges.remove((now_node, next_node))
#         passed_interdicted_straight_edges.append((now_node, next_node))

#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_straight_edge, width=3.0, edge_color='#191970')
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_straight_edge, width=3.0, style='--', edge_color='#191970')
#     arc_rad = 0.29
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

#     edge_weights = nx.get_edge_attributes(graph,'weight')
#     edge_plus = nx.get_edge_attributes(graph,'plus')
#     curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
#     interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in interdicted_curved_edges}
#     straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
#     interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in interdicted_straight_edges}
#     pass_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in pass_curved_edge}
#     pass_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in pass_interdicted_curved_edge}
#     pass_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in pass_straight_edge}
#     pass_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in pass_interdicted_straight_edge}
#     my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
#     my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
#     my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_curved_edge_labels,rotate=False,rad = arc_rad)
#     my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
#     nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
#     nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
#     nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_straight_edge_labels,rotate=False)
#     nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_straight_edge_labels,rotate=False)
#     plt.show()

#     if len(pass_curved_edge) == 1:
#         curved_edges.append(pass_curved_edge.pop())
#     elif len(pass_interdicted_curved_edge) == 1:
#         interdicted_curved_edges.append(pass_interdicted_curved_edge.pop())
#     elif len(pass_straight_edge) == 1:
#         straight_edges.append(pass_straight_edge.pop())
#     else:
#         interdicted_straight_edges.append(pass_interdicted_straight_edge.pop())

#     now_S = next_S
#     now_node = next_node

# print('最適解のコスト')
# print(z_star[()][s])

# for i in passed_curved_edges:
#     if i in curved_edges:
#         curved_edges.remove(i)
# for i in passed_interdicted_curved_edges:
#     if i in interdicted_curved_edges:
#         interdicted_curved_edges.remove(i)
# for i in passed_straight_edges:
#     if i in straight_edges:
#         straight_edges.remove(i)
# for i in passed_interdicted_straight_edges:
#     if i in interdicted_straight_edges:
#         interdicted_straight_edges.remove(i)

# fig, ax = plt.subplots()
# nx.draw_networkx_nodes(graph, pos, ax = ax)
# nx.draw_networkx_labels(graph, pos, ax = ax)
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_straight_edges, width=3.0, edge_color='#191970')
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_straight_edges, width=3.0, style='--', edge_color='#191970')
# arc_rad = 0.29
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
# nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

# edge_weights = nx.get_edge_attributes(graph,'weight')
# edge_plus = nx.get_edge_attributes(graph,'plus')
# curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
# interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in interdicted_curved_edges}
# straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
# interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in interdicted_straight_edges}
# passed_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in passed_curved_edges}
# passed_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in passed_interdicted_curved_edges}
# passed_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in passed_straight_edges}
# passed_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in passed_interdicted_straight_edges}
# my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
# my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
# my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_curved_edge_labels,rotate=False,rad = arc_rad)
# my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
# nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
# nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
# nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_straight_edge_labels,rotate=False)
# nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_straight_edge_labels,rotate=False)
# plt.show()

# print('straight_edges')
# print(straight_edges)
# print('interdicted_straight_edges')
# print(interdicted_straight_edges)
# print('pass_straight_edge')
# print(pass_straight_edge)
# print('pass_interdicted_straight_edge')
# print(pass_interdicted_straight_edge)
# print('passed_straight_edges')
# print(passed_straight_edges)
# print('passed_interdicted_straight_edges')
# print(passed_interdicted_straight_edges)
# print('curved_edges')
# print(curved_edges)
# print('interdicted_curved_edges')
# print(interdicted_curved_edges)
# print('pass_curved_edge')
# print(pass_curved_edge)
# print('pass_interdicted_curved_edge')
# print(pass_interdicted_curved_edge)
# print('passed_curved_edges')
# print(passed_curved_edges)
# print('passed_interdicted_curved_edges')
# print(passed_interdicted_curved_edges)

# 5 9 0 4 2 0 1 2 2 0 2 6 2 1 2 3 4 1 3 1 2 1 4 4 8 2 1 4 2 2 4 2 4 3 1 2 4 3 4 4 4
# 5 6 0 4 2 0 1 1 2 0 2 1 2 1 4 1 10 2 3 1 2 2 4 3 1 3 4 1 10
# 7 12 0 6 5 0 5 2 7 1 3 1 9 2 3 4 3 2 6 4 6 3 4 2 1 3 5 2 5 3 6 4 9 4 0 1 5 4 2 2 7 4 3 2 2 5 2 4 10 5 4 4 1
# 6 12 0 5 4 0 3 3 3 0 4 1 10 1 4 5 4 1 5 3 7 2 0 2 5 2 1 3 5 2 4 2 5 2 5 4 6 3 2 2 10 3 4 4 2 4 2 1 3 4 5 3 9
