from heapq import heappush, heappop
from collections import defaultdict
import itertools
import copy

import networkx as nx

import my_modules

def get_DSPI_at_most(input_list):
    INF = 10 ** 9
    # input_list = list(map(int, input().split()))
    n, m, s, t, budget = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
    # graph = nx.DiGraph()
    # edge_list = []
    G = [[] for i in range(n)]
    global G_rev
    G_rev = [[] for i in range(n)]
    # A, Arcs = (), ()
    list_A, list_Arcs = [], []

    for i in range(m):
        a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
        # edge_list.append((a, b, {'weight':w, 'plus':p}))
        G[a].append([a, b, w, p])
        G_rev[b].append([a, w])
        # A += ((a, b, w, p), )
        # Arcs += ((a, b), )
        list_A.append((a, b, w, p))
        list_Arcs.append((a, b))

    for i in range(n):
        G[i].sort()
        G_rev[i].sort()
    list_A.sort()
    list_Arcs.sort()
    tuple_A = my_modules.list_to_tuple(list_A)
    tuple_Arcs = my_modules.list_to_tuple(list_Arcs)

    # z_star, z_barの定義 (キー : tuple, 要素 : list)
    z_star = defaultdict(list)
    z_bar = defaultdict(list)
    next_z_star = defaultdict(list)
    next_z_bar = defaultdict(list)
    input_list

    # 何も阻止されていない場合で逆向きのダイクストラを行い, 前にいた頂点のリストとコストを出力
    prev, d = my_modules.dijkstra(n - 1, n, INF, G_rev)

    # 経路を復元して, all-to-t-shortest path tree Tを作成
    T = my_modules.get_T(n, prev)

    # |S|==bを満たすSの集合cを定義する
    c = list(itertools.combinations(tuple_A, budget))

    # |S|==bの各Sについて, Tの辺を含むか判定し, 含む場合はGを定義してダイクストラを行う.
    for S in c:
        G_rev = [[] for i in range(n)]
        flag = False
        for j in tuple_A:
            if j in S:
                if (j[0], j[1]) in T: # Tに含まれているかどうかの判定
                    flag = True
                G_rev[j[1]].append((j[0], j[2] + j[3]))
            else:
                G_rev[j[1]].append((j[0], j[2]))
        if flag:
            prev, list_z_star_S = my_modules.dijkstra(n - 1, n, INF, G_rev)
            # print('S, z_star_S, prev')
            # print(S, list_z_star_S, prev)
            z_star[S] = list_z_star_S
            next_z_star[S] = [[(), -1] for i in range(n)]
            # print(next_z_star[S])
            for i in range(n - 1):
                next_z_star[S][i] = (S, prev[i])
    # print('z_star')
    # print(z_star)

    # DP-DSPIの2行目
    for k in range(budget-1, -1, -1):
        # DP-DSPIの3行目
        c = list(itertools.combinations(tuple_A, k))
        for S in c:
            z_star[S] = [0] * n # continueしたものは0が代入される
            # next_z_bar[S] = [[(), -1] for i in range(n)]
            next_z_star[S] = [[(), -1] for i in range(n)]
            list_S = my_modules.tuple_to_list(S)
            # print('list_S')
            # print(list_S)
            for i in range(n-1):
                Arcs = copy.deepcopy(list_A)
                # Arcsの要素のうち, すでにSに含まれているものを削除
                for item in list_S:
                    if item in Arcs:
                        Arcs.remove(item)
                num = len(Arcs)
                # print('list_S, i, Arcs')
                # print(list_S, i, Arcs)
                # 阻止できる辺が存在する場合
                if num > 0:
                    # 全探索でS_primeを全列挙する
                    max_val = 0
                    new_max_S = tuple()
                    new_max_i = -1
                    # for j in range(1, 2**num):
                    #     new_S = copy.deepcopy(list_S)
                    #     for l in range(num):
                    #         if j >> l & 1:
                    #             # S_primeに追加する
                    #             new_S.append(Arcs[l])
                    # for j in range(num):
                    #     new_S = copy.deepcopy(list_S)
                    #     new_S.append(Arcs[j])
                    for j in range(num):
                        new_S = copy.deepcopy(list_S)
                        new_S.append(Arcs[j])
                        # 予算を超えていたら次のnew_Sを探索
                        if  len(new_S) > budget:
                            continue
                        new_S.sort()
                        new_S = my_modules.list_to_tuple(new_S)
                        # z_star[new_S]が存在するかの判定
                        if z_star[new_S] == []:
                            # print('含まれないの発見')
                            # print(S, i, new_S)
                            continue
                        min_val = INF
                        # new_min_S = new_S
                        for list_arc in G[i]:
                            # c_tildeの更新
                            arc = my_modules.list_to_tuple(list_arc)
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
                    z_star[S][i] = max_val
                    next_z_star[S][i] = (new_max_S, new_max_i)
                # 阻止できる辺が存在しない場合
                else:
                    min_val = INF
                    for list_arc in G[i]:
                        # c_tildeの更新
                        arc = my_modules.list_to_tuple(list_arc)
                        if arc in S:
                            c_tilde = arc[2] + arc[3]
                        else:
                            c_tilde = arc[2]
                        if min_val > z_star[S][arc[1]] + c_tilde:
                            min_val = z_star[S][arc[1]] + c_tilde
                            new_min_i = arc[1]
                    z_star[S][i] = min_val
                    next_z_star[S][i] = (S, new_min_i)
    return z_star, next_z_star, tuple_Arcs, tuple_A

