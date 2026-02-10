import os
import pickle
import time
import torch
import networkx as nx
from tqdm import tqdm
from struct_entropy import StructEntropy


def load_graph(file_path):
    with open(file_path, 'rb') as file:
        nx_graph = pickle.load(file)
    return nx_graph


def vanilla_2D_SE_mini(weighted_edges=None, file_path=None):
    '''
    vanilla (greedy) 2D SE minimization
    '''

    if file_path is not None:
        graph_tag = load_graph(file_path)
        g = graph_tag[0]
    elif weighted_edges is not None:
        g = nx.Graph()
        g.add_weighted_edges_from(weighted_edges)
    else:
        graph_tag = load_graph("/share/project/duli/tag_community_discovery/graph_partion.pkl")
        g = graph_tag[0]

    times = {}

    t0 = time.perf_counter()
    seg = StructEntropy(g)
    seg.init_division()
    # seg.show_division()
    SE1D = seg.calc_1dSE()
    times['calc_1dSE'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seg.update_struc_data()
    # seg.show_struc_data()
    times['update_struc_data'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seg.update_struc_data_2d()
    # seg.show_struc_data_2d()
    times['update_struc_data_2d'] = time.perf_counter() - t0
    initial_SE2D = seg.calc_2dSE()

    t0 = time.perf_counter()
    seg.update_division_MinSE()
    times['update_division_MinSE'] = time.perf_counter() - t0
    communities = seg.division

    t0 = time.perf_counter()
    minimized_SE2D = seg.calc_2dSE()
    times['calc_2dSE_minimized'] = time.perf_counter() - t0

    print(times)

    return SE1D, initial_SE2D, minimized_SE2D, communities


if __name__ == '__main__':
    g_nx = load_graph(
        "/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/skill_graph_zh_ability.pkl")  # /share/project/duli/tag_community_discovery/full_graph_130w_dedup.pkl
    # g_nx = graph_tag[0]

    # --- Convert NetworkX graph to PyTorch tensors for GPU version ---
    edges_list = []
    weights_list = []
    relations_list = []

    # Create a mapping from original node names to contiguous integer IDs
    node_mapping = {node: i for i, node in enumerate(g_nx.nodes())}
    reverse_node_mapping = {i: node for node, i in node_mapping.items()}

    # Extract edges, weights, and relations
    for u, v, data in g_nx.edges(data=True):
        edges_list.append([node_mapping[u], node_mapping[v]])
        weights_list.append(data.get('weight', 0.0))
        relations_list.append(data.get('relation', 'related to'))  # undirected graph
    print("Total Nodes and Edges of Graph: ", len(g_nx.nodes()), len(g_nx.edges()))

    # Convert to PyTorch tensors and move to the selected device
    edges_tensor = torch.tensor(edges_list, dtype=torch.long, device="cuda")
    weights_tensor = torch.tensor(weights_list, dtype=torch.float, device="cuda")

    # Instantiate StructEntropy with edges, weights, and relations
    codingTree = StructEntropy(edges=edges_tensor, weights=weights_tensor, relations=relations_list)

    # Minimum Coding Tree Optimization
    print("Starting K-dimension Minimum encoding tree optimization")
    start_time = time.time()
    community_tree = codingTree.find_k_dim_entropy_tree(k_dim=4)
    end_time = time.time()  # End timing
    print("Minimum encoding tree optimization Done!")
    print(f"[Time] Minimum encoding tree optimization: {end_time - start_time:.4f} seconds")


    def print_tree(node, level=0):
        indent = "  " * level
        print(f"{indent}Community (Level {level}): {node.node_ids}")
        for child in node.children:
            print_tree(child, level + 1)


    print("\n--- Final Hierarchical Community Encoding Tree ---")
    print_tree(community_tree)

    # First, calculate K Dimension SE for all nodes on the GPU
    print("\nCalculating node K-Dim SE on subgraph...")

    entropy_value = codingTree.calc_se_from_tree()
    print(f"K-Dimension Struct Entropy: {entropy_value:.4f}")

    # --- 组合泛化任务: Find max conditional entropy set for each ability ---
    print("\n--- Starting Combination Generalization Task ---")

    # 结果的保存路径
    batch_size = 100
    set_size = 7  # 组合数k
    beam_width = 100
    print(f"Set size: {set_size}, Beam width: {beam_width}")

    output_dir = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/agent/k={set_size}/"
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    base_filename = "optimal_combinations"

    temp_comb_list = []
    batch_counter = 0

    all_ability_labels = list(reverse_node_mapping.values())

    for i in tqdm(range(codingTree.num_nodes), desc="Finding optimal combinations"):
        start_node = i
        # find_max_chain_conditional_entropy_set 函数需要一个列表作为x_set
        x_set = [start_node]

        top_k_results = codingTree.find_max_global_conditional_entropy_sets(
            x_set,
            set_size=set_size - 1,  # 目标 = k -1
            beam_width=beam_width
        )
        for final_cond_se, y_set in top_k_results:
            # 将结果从节点ID转换回原始的ability label
            start_label = reverse_node_mapping[start_node]
            y_labels = [reverse_node_mapping[node_id] for node_id in y_set]

            # 存储每一个结果
            result = {
                "start_ability": start_label,
                "optimal_combination": y_labels,
                "max_conditional_se": final_cond_se
            }
            temp_comb_list.append(result)

        if (i + 1) % batch_size == 0 or (i + 1) == codingTree.num_nodes:
            # 达到 batch_size 或到达最后一个节点
            batch_counter += 1
            file_name = f"{output_dir}{base_filename}_{batch_counter}.pkl"

            with open(file_name, 'wb') as f:
                pickle.dump(temp_comb_list, f)
            print(f"\nBatch {batch_counter}: Results from node {i - len(temp_comb_list) // len(top_k_results) + 1} "
                  f"to {i} successfully saved to: {file_name}")
            temp_comb_list = []

