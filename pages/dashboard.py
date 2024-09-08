import streamlit as st
import matplotlib.pyplot as plt

import networkx as nx

from services import DSPI_free, DSPI_at_most, DSPI_at_least, DSPI_exactly, my_networkx as my_nx

def graph_drawing_cost_increase(input_list, z_star, next_z_star, tuple_Arcs, tuple_A):
    now_node = 0
    now_S = tuple()
    now_cost = 0
    # グラフの描画準備
    graph = nx.DiGraph()
    edge_list = []
    n, m, t = input_list[0], input_list[1], input_list[3]
    for i in range(m):
        a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
        edge_list.append((a, b, {'weight':w, 'plus':p}))

    graph.add_edges_from(edge_list)
    pos  = nx.kamada_kawai_layout(graph)
    for i in range(n):
        pos[i][0] = pos[i][0] * (-1)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax = ax)
    nx.draw_networkx_labels(graph, pos, ax = ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.29
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(graph,'weight')
    edge_plus = nx.get_edge_attributes(graph,'plus')
    curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
    straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad = arc_rad)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
    # plt.show()
    st.pyplot(fig)
    interdicted_curved_edges = []
    interdicted_straight_edges = []
    passed_curved_edges = []
    passed_interdicted_curved_edges = []
    passed_straight_edges = []
    passed_interdicted_straight_edges = []
    while now_node != t:
        next_S_and_node = next_z_star[now_S][now_node]
        next_S = next_S_and_node[0]
        next_node = next_S_and_node[1]
        now_cost += z_star[now_S][now_node] - z_star[next_S][next_node]
        # print('阻止集合')
        # print(next_S)
        # print('今回の移動')
        # print(str(now_node) + '→' + str(next_node))
        # print('ここまでの総コスト')
        # print(now_cost)
        # グラフの描画
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(graph, pos, ax = ax)
        nx.draw_networkx_labels(graph, pos, ax = ax)

        # curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
        # straight_edges = list(set(graph.edges()) - set(curved_edges))

        # interdicted_curved_edges = []
        for i in range(len(curved_edges)):
            res = tuple_Arcs.index(curved_edges[i])
            if tuple_A[res] in next_S:
                interdicted_curved_edges.append(curved_edges[i])
        for i in interdicted_curved_edges:
            if i in curved_edges:
                curved_edges.remove(i)

        # interdicted_straight_edges = []
        for i in range(len(straight_edges)):
            res = tuple_Arcs.index(straight_edges[i])
            if tuple_A[res] in next_S:
                interdicted_straight_edges.append(straight_edges[i])
        for i in interdicted_straight_edges:
            if i in straight_edges:
                straight_edges.remove(i)

        pass_curved_edge = []
        pass_interdicted_curved_edge = []
        pass_straight_edge = []
        pass_interdicted_straight_edge = []
        if (now_node, next_node) in curved_edges:
            pass_curved_edge.append((now_node, next_node))
            curved_edges.remove((now_node, next_node))
            passed_curved_edges.append((now_node, next_node))
        elif (now_node, next_node) in interdicted_curved_edges:
            pass_interdicted_curved_edge.append((now_node, next_node))
            interdicted_curved_edges.remove((now_node, next_node))
            passed_interdicted_curved_edges.append((now_node, next_node))
        elif (now_node, next_node) in straight_edges:
            pass_straight_edge.append((now_node, next_node))
            straight_edges.remove((now_node, next_node))
            passed_straight_edges.append((now_node, next_node))
        else:
            pass_interdicted_straight_edge.append((now_node, next_node))
            interdicted_straight_edges.remove((now_node, next_node))
            passed_interdicted_straight_edges.append((now_node, next_node))

        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_straight_edge, width=3.0, edge_color='#191970')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_straight_edge, width=3.0, style='--', edge_color='#191970')
        arc_rad = 0.29
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

        edge_weights = nx.get_edge_attributes(graph,'weight')
        edge_plus = nx.get_edge_attributes(graph,'plus')
        curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
        interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in interdicted_curved_edges}
        straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
        interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in interdicted_straight_edges}
        pass_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in pass_curved_edge}
        pass_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in pass_interdicted_curved_edge}
        pass_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in pass_straight_edge}
        pass_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in pass_interdicted_straight_edge}
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_straight_edge_labels,rotate=False)
        # plt.show()
        st.pyplot(fig)

        if len(pass_curved_edge) == 1:
            curved_edges.append(pass_curved_edge.pop())
        elif len(pass_interdicted_curved_edge) == 1:
            interdicted_curved_edges.append(pass_interdicted_curved_edge.pop())
        elif len(pass_straight_edge) == 1:
            straight_edges.append(pass_straight_edge.pop())
        else:
            interdicted_straight_edges.append(pass_interdicted_straight_edge.pop())

        now_S = next_S
        now_node = next_node

    # print('最適解のコスト')
    # print(z_star[()][s])

    for i in passed_curved_edges:
        if i in curved_edges:
            curved_edges.remove(i)
    for i in passed_interdicted_curved_edges:
        if i in interdicted_curved_edges:
            interdicted_curved_edges.remove(i)
    for i in passed_straight_edges:
        if i in straight_edges:
            straight_edges.remove(i)
    for i in passed_interdicted_straight_edges:
        if i in interdicted_straight_edges:
            interdicted_straight_edges.remove(i)

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax = ax)
    nx.draw_networkx_labels(graph, pos, ax = ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_straight_edges, width=3.0, edge_color='#191970')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_straight_edges, width=3.0, style='--', edge_color='#191970')
    arc_rad = 0.29
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

    edge_weights = nx.get_edge_attributes(graph,'weight')
    edge_plus = nx.get_edge_attributes(graph,'plus')
    curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in curved_edges}
    interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in interdicted_curved_edges}
    straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in straight_edges}
    interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in interdicted_straight_edges}
    passed_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in passed_curved_edges}
    passed_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) + ' (' +  str(edge_plus[edge]) + ')' for edge in passed_interdicted_curved_edges}
    passed_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in passed_straight_edges}
    passed_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) + ' (' + str(edge_plus[edge]) + ')' for edge in passed_interdicted_straight_edges}
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_straight_edge_labels,rotate=False)
    # plt.show()
    st.pyplot(fig)

def graph_drawing_remove_arcs(input_list, z_star, next_z_star, tuple_Arcs, tuple_A):
    now_node = 0
    now_S = tuple()
    now_cost = 0
    # グラフの描画準備
    graph = nx.DiGraph()
    edge_list = []
    n, m, t = input_list[0], input_list[1], input_list[3]
    for i in range(m):
        a, b, w, p = input_list[i*4+5], input_list[i*4+6], input_list[i*4+7], input_list[i*4+8]
        edge_list.append((a, b, {'weight':w, 'plus':p}))

    graph.add_edges_from(edge_list)
    pos  = nx.kamada_kawai_layout(graph)
    for i in range(n):
        pos[i][0] = pos[i][0] * (-1)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax = ax)
    nx.draw_networkx_labels(graph, pos, ax = ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.29
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(graph,'weight')
    # edge_plus = nx.get_edge_attributes(graph,'plus')
    curved_edge_labels = {edge: str(edge_weights[edge]) for edge in curved_edges}
    straight_edge_labels = {edge: str(edge_weights[edge]) for edge in straight_edges}
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad = arc_rad)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
    # plt.show()
    st.pyplot(fig)
    interdicted_curved_edges = []
    interdicted_straight_edges = []
    passed_curved_edges = []
    passed_interdicted_curved_edges = []
    passed_straight_edges = []
    passed_interdicted_straight_edges = []
    while now_node != t:
        next_S_and_node = next_z_star[now_S][now_node]
        next_S = next_S_and_node[0]
        next_node = next_S_and_node[1]
        now_cost += z_star[now_S][now_node] - z_star[next_S][next_node]
        # print('阻止集合')
        # print(next_S)
        # print('今回の移動')
        # print(str(now_node) + '→' + str(next_node))
        # print('ここまでの総コスト')
        # print(now_cost)
        # グラフの描画
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(graph, pos, ax = ax)
        nx.draw_networkx_labels(graph, pos, ax = ax)

        # curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
        # straight_edges = list(set(graph.edges()) - set(curved_edges))

        # interdicted_curved_edges = []
        for i in range(len(curved_edges)):
            res = tuple_Arcs.index(curved_edges[i])
            if tuple_A[res] in next_S:
                interdicted_curved_edges.append(curved_edges[i])
        for i in interdicted_curved_edges:
            if i in curved_edges:
                curved_edges.remove(i)

        # interdicted_straight_edges = []
        for i in range(len(straight_edges)):
            res = tuple_Arcs.index(straight_edges[i])
            if tuple_A[res] in next_S:
                interdicted_straight_edges.append(straight_edges[i])
        for i in interdicted_straight_edges:
            if i in straight_edges:
                straight_edges.remove(i)

        pass_curved_edge = []
        pass_interdicted_curved_edge = []
        pass_straight_edge = []
        pass_interdicted_straight_edge = []
        if (now_node, next_node) in curved_edges:
            pass_curved_edge.append((now_node, next_node))
            curved_edges.remove((now_node, next_node))
            passed_curved_edges.append((now_node, next_node))
        elif (now_node, next_node) in interdicted_curved_edges:
            pass_interdicted_curved_edge.append((now_node, next_node))
            interdicted_curved_edges.remove((now_node, next_node))
            passed_interdicted_curved_edges.append((now_node, next_node))
        elif (now_node, next_node) in straight_edges:
            pass_straight_edge.append((now_node, next_node))
            straight_edges.remove((now_node, next_node))
            passed_straight_edges.append((now_node, next_node))
        else:
            pass_interdicted_straight_edge.append((now_node, next_node))
            interdicted_straight_edges.remove((now_node, next_node))
            passed_interdicted_straight_edges.append((now_node, next_node))

        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_straight_edge, width=3.0, edge_color='#191970')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_straight_edge, width=3.0, style='--', edge_color='#191970')
        arc_rad = 0.29
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=pass_interdicted_curved_edge, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

        edge_weights = nx.get_edge_attributes(graph,'weight')
        # edge_plus = nx.get_edge_attributes(graph,'plus')
        curved_edge_labels = {edge: str(edge_weights[edge]) for edge in curved_edges}
        interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in interdicted_curved_edges}
        straight_edge_labels = {edge: str(edge_weights[edge]) for edge in straight_edges}
        interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in interdicted_straight_edges}
        pass_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in pass_curved_edge}
        pass_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in pass_interdicted_curved_edge}
        pass_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in pass_straight_edge}
        pass_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in pass_interdicted_straight_edge}
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_curved_edge_labels,rotate=False,rad = arc_rad)
        my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_straight_edge_labels,rotate=False)
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=pass_interdicted_straight_edge_labels,rotate=False)
        # plt.show()
        st.pyplot(fig)

        if len(pass_curved_edge) == 1:
            curved_edges.append(pass_curved_edge.pop())
        elif len(pass_interdicted_curved_edge) == 1:
            interdicted_curved_edges.append(pass_interdicted_curved_edge.pop())
        elif len(pass_straight_edge) == 1:
            straight_edges.append(pass_straight_edge.pop())
        else:
            interdicted_straight_edges.append(pass_interdicted_straight_edge.pop())

        now_S = next_S
        now_node = next_node

    # print('最適解のコスト')
    # print(z_star[()][s])

    for i in passed_curved_edges:
        if i in curved_edges:
            curved_edges.remove(i)
    for i in passed_interdicted_curved_edges:
        if i in interdicted_curved_edges:
            interdicted_curved_edges.remove(i)
    for i in passed_straight_edges:
        if i in straight_edges:
            straight_edges.remove(i)
    for i in passed_interdicted_straight_edges:
        if i in interdicted_straight_edges:
            interdicted_straight_edges.remove(i)

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax = ax)
    nx.draw_networkx_labels(graph, pos, ax = ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_straight_edges, width=2.0, style='--')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_straight_edges, width=3.0, edge_color='#191970')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_straight_edges, width=3.0, style='--', edge_color='#191970')
    arc_rad = 0.29
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=2.0, style='--')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, edge_color='#191970')
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=passed_interdicted_curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', width=3.0, style='--', edge_color='#191970')

    edge_weights = nx.get_edge_attributes(graph,'weight')
    # edge_plus = nx.get_edge_attributes(graph,'plus')
    curved_edge_labels = {edge: str(edge_weights[edge]) for edge in curved_edges}
    interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in interdicted_curved_edges}
    straight_edge_labels = {edge: str(edge_weights[edge]) for edge in straight_edges}
    interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in interdicted_straight_edges}
    passed_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in passed_curved_edges}
    passed_interdicted_curved_edge_labels = {edge: str(edge_weights[edge]) for edge in passed_interdicted_curved_edges}
    passed_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in passed_straight_edges}
    passed_interdicted_straight_edge_labels = {edge: str(edge_weights[edge]) for edge in passed_interdicted_straight_edges}
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_curved_edge_labels,rotate=False,rad = arc_rad)
    my_nx.my_draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_curved_edge_labels,rotate=False,rad = arc_rad)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=interdicted_straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_straight_edge_labels,rotate=False)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=passed_interdicted_straight_edge_labels,rotate=False)
    # plt.show()
    st.pyplot(fig)

def display():
    st.header('Input Graph Data')
    # Warningの非表示
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # 入力の受け取り
    input_text = st.text_input('Graph Data')
    input_list = list(map(int, input_text.split()))
    option_type = st.selectbox('Select the type of Interdiction', ['Cost increase', 'Remove arcs'])
    option_constraints = st.selectbox('Select constraint', ['free', 'at most', 'at least', 'exactly'])

    # DSPIの実行とグラフ描画を実行するボタンを表示する
    if st.button('Run DSPI'):
        if option_type == 'Cost increase':
            if option_constraints == 'free':
                z_star, next_z_star, tuple_Arcs, tuple_A = DSPI_free.get_DSPI_free(input_list)
                graph_drawing_cost_increase(input_list, z_star, next_z_star, tuple_Arcs, tuple_A)
            elif option_constraints == 'at most':
                z_star, next_z_star, tuple_Arcs, tuple_A = DSPI_at_most.get_DSPI_at_most(input_list)
                graph_drawing_cost_increase(input_list, z_star, next_z_star, tuple_Arcs, tuple_A)
            elif option_constraints == 'at least':
                z_star, next_z_star, tuple_Arcs, tuple_A = DSPI_at_least.get_DSPI_at_least(input_list)
                graph_drawing_cost_increase(input_list, z_star, next_z_star, tuple_Arcs, tuple_A)
            else:
                z_star, next_z_star, tuple_Arcs, tuple_A = DSPI_exactly.get_DSPI_exactly(input_list)
                graph_drawing_cost_increase(input_list, z_star, next_z_star, tuple_Arcs, tuple_A)
        else:
            if option_constraints == 'free':
                '作成中'
            elif option_constraints == 'at most':
                '作成中'
            elif option_constraints == 'at least':
                '作成中'
            else:
                '作成中'
