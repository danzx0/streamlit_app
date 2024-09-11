import random
import copy
import pulp

from. import my_modules

def run_arc_remove_b_simulation(epoch, a, b):
    INF = 10**9
    # epoch = 10000
    averages = []
    # input_lists = []
    for n in range(a, b):
        res = 0
        # パラメータ
        Delta = ['delta_' + str(i) for i in range(n)]
        # print(Delta)
        Problem_names = ['Problem_' + str(i) for i in range(n)]
        # nodes = [str(i) for i in range(n)]
        for uchigawa in range(epoch):
            m = random.randint(n-1, n*(n-1) - (n-1))
            s = 0
            t = n-1
            budget = random.randint(1, 5)

            X = [i for i in range(n-1)]
            Y = [n-1]
            G = [[] for i in range(n)]
            G_node = [[] for i in range(n)]
            Arcs = []
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
                Arcs.append((x, y))
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
                Arcs.append((a, b))
                res_Arcs[a] += 1
                if res_Arcs[a] == n-1:
                    not_node.append(a)

            for i in range(n):
                G[i].sort()
                G_node[i].sort()
            Arcs.sort()
            Arcs = my_modules.list_to_tuple(Arcs)

            input_list = [n, m, s, t, budget]
            for i in range(n):
                for j in range(len(G[i])):
                    input_list.append(i)
                    input_list.append(G[i][j][0])
                    input_list.append(G[i][j][1])
                    input_list.append(G[i][j][2])
            # print(*input_list)
            # input_lists.append(input_list)

            # n, m, s, t, budget = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
            # graph = nx.DiGraph()
            # edge_list = []
            G = [[] for i in range(n)]
            G_rev = [[] for i in range(n)]
            # G_interdicted_reverse = [[] for i in range(n)]
            # A, Arcs = (), ()
            list_A = []

            for i in range(m):
                a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
                # edge_list.append((a, b, {'weight':w, 'plus':p}))
                G[a].append([a, b, w, p])
                G_rev[b].append([a, w])
                # G_interdicted_reverse[b].append([a, w + p])
                # A += ((a, b, w, p), )
                # Arcs += ((a, b), )
                list_A.append((a, b, w, p))
                # list_Arcs.append((a, b))

            for i in range(n):
                G[i].sort()
                G_rev[i].sort()
                # G_interdicted_reverse[i].sort()
            list_A.sort()
            # list_Arcs.sort()
            tuple_A = my_modules.list_to_tuple(list_A)
            # tuple_Arcs = my_modules.list_to_tuple(list_Arcs)

            # 何も阻止されていない場合で逆向きのダイクストラを行い, 前にいた頂点のリストとコストを出力
            prev, d = my_modules.dijkstra(n-1, n, INF, G_rev)
            # print('Sが空集合のときのprev')
            # print(prev)
            # print('Sが空集合のときのtからのコスト')
            # print(d)

            # 経路を復元して, all-to-t-shortest path tree Tを作成
            # T = get_T(prev)
            # print('T')
            # print(T)

            # すべての辺が阻止された場合で逆向きのダイクストラを行い, 前にいた頂点のリストとコストを出力
            # prev, d_interdicted = dijkstra_interdicted(n-1, n)
            # print('すべての辺が阻止されたときのprev')
            # print(prev)
            # print('すべての辺が阻止されたときのtからのコスト')
            # print(d_interdicted)

            # パラメータ
            # Delta = ['delta_' + str(i) for i in range(n)]
            # print(Delta)
            Z = ['z_' + str(i) for i in tuple_A]
            # print(Z)
            # Problem_names = ['Problem_' + str(i) for i in range(n)]
            # Problem_answers = defaultdict(list)
            # nodes = [str(i) for i in range(n)]
            # Problem_answers['Problems  '] = ['Status', 'delta', *nodes, 'z', *tuple_Arcs]
            # print(Problem_answers)
            delta_values = []

            for i in range(n):
                # 最大化
                problem = pulp.LpProblem(Problem_names[i], pulp.LpMaximize)

                # 変数定義
                delta = pulp.LpVariable.dicts('delta', Delta, cat='Continuous')
                z = pulp.LpVariable.dicts('z', Z, cat='Binary')

                # 目的関数
                start = 'delta_' + str(i)
                problem += delta[start] - delta[Delta[-1]]

                # 制約条件
                for j in range(len(tuple_A)):
                    problem += delta[Delta[tuple_A[j][0]]] <= delta[Delta[tuple_A[j][1]]] + tuple_A[j][2] + tuple_A[j][3] * z[Z[j]]

                problem += delta[Delta[-1]] == 0

                problem += pulp.lpSum(z) == budget

                # 求解
                status = problem.solve()
                delta_values.append(delta[Delta[i]].value())
                # Problem_answers[Problem_names[i]] = []
                # print(Problem_names[i])
                # print('Status', pulp.LpStatus[status])
                # Problem_answers[Problem_names[i]].append(pulp.LpStatus[status])
                # Problem_answers[Problem_names[i]].append('delta')
                # for item in Delta:
                #     print('delta[', item, '] = ',  delta[item].value())
                #     Problem_answers[Problem_names[i]].append(delta[item].value())
                # Problem_answers[Problem_names[i]].append('z')
                # for item in Z:
                #     print('z[', item, '] = ',  z[item].value())
                #     Problem_answers[Problem_names[i]].append(z[item].value())
                # print('目的関数値 = ', problem.objective.value())
                # Problem_answers[Problem_names[i]].append('目的関数値 = ')
                # Problem_answers[Problem_names[i]].append(problem.objective.value())

            # for i in Problem_answers.items():
            #     print(i)

            removed_edge = []

            for i in range(n):
                A_i = copy.deepcopy(G[i])
                # print('A_i')
                # print(A_i)
                for j in range(len(A_i)):
                    # すべての辺が阻止されていない場合の「i → jのコスト + j → tのダイクストラの結果」を計算
                    not_interdicted_cost = A_i[j][2] + d[A_i[j][1]]
                    # b本の辺を最適に阻止した場合の「i → tのコスト」と比較
                    if not_interdicted_cost > delta_values[i]:
                        # print('削除前')
                        # print(list_A)
                        # print(list_Arcs)
                        # print(G)
                        # i → jを削除
                        # print('削除する辺 ', i, '→', A_i[j][1])
                        removed_edge.append([i, A_i[j][1]])
                        list_A.remove(tuple(A_i[j]))
                        # list_Arcs.remove((A_i[j][0], A_i[j][1]))
                        G[i].remove(A_i[j])
                        # print('削除後')
                        # print(list_A)
                        # print(list_Arcs)
                        # print(G)
            # print('削除する辺')
            # print(removed_edge)
            xx = len(removed_edge)/m
            # print(n, uchigawa)
            # print(xx)
            # wariai.append(xx)
            res += xx
        ave = res/epoch
        # print(str(n) + 'での削除割合の平均')
        # print(ave)
        averages.append(ave)
    # print('ノード数ごとの削除割合の平均')
    # print(averages)
    return averages
