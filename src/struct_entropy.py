# Copyright 2023 Weiyifan <weiyifan@buaa.edu.cn>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-20.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import math
import itertools
import collections
import networkx as nx
from tqdm import tqdm

# set device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"


class CommunityNode:
    def __init__(self, node_ids, parent=None):
        self.node_ids = node_ids
        self.parent = parent
        self.children = []
        self.volume = 0.0
        self.cut = 0.0
        self.se_term = 0.0
        # Use a unique ID for each community node for easier lookup
        self.ID = hash(tuple(sorted(node_ids)))


class StructEntropy:
    def __init__(self, edges: torch.Tensor, weights: torch.Tensor, relations: list = None):
        self.edges = edges.to(device)
        self.weights = weights.to(device)
        self.num_nodes = int(self.edges.max() + 1) if self.edges.numel() > 0 else 0
        self.relations = relations
        self.node_to_comm = torch.arange(self.num_nodes, device=device)
        self.community_tree = None
        self.graph_node_to_leaf_map = {}  # Map graph nodes to the leaf nodes of coding tree
        self.vol = self._calc_graph_volume()
        self.degrees = self._get_degrees(self.edges, self.weights, self.num_nodes)
        self.node_se = None
        self.all_community_nodes = {}  # Store all community nodes by their ID

    def _calc_graph_volume(self):
        """
        Calculates the sum of all node degrees in the Graph G (2m in the theory).
        (Fix: Ensures the volume is 2 * sum of weights for undirected graph theory)
        """
        if self.weights.numel() == 0:
            return 0.0
        # Correctly return 2m
        return 2 * torch.sum(self.weights).item()

    def _get_degrees(self, edges, weights, num_nodes):
        """Calculates the degree of each node in the subgraph."""
        degrees = torch.zeros(num_nodes, dtype=weights.dtype, device=device)
        degrees.scatter_add_(0, edges[:, 0], weights)
        degrees.scatter_add_(0, edges[:, 1], weights)
        return degrees

    def _get_community_properties(self, edges, weights, node_to_comm, num_nodes, graph_vol):
        """
        NP-hard: Greedy merging is used to approximate the global optimal solution.
        Calculates the volume and cut for each community in a given partition.

        Args:
            edges (torch.Tensor): Edges of the subgraph.
            weights (torch.Tensor): Weights of the subgraph edges.
            node_to_comm (torch.Tensor): Mapping from node IDs to community IDs.
            num_nodes (int): Number of nodes in the subgraph.
            graph_vol (float): Total volume (sum of degrees) of the subgraph.

        Returns:
            tuple: A tuple containing community IDs, volumes, cuts, and SE terms.
        """
        comm_ids, node_to_comm_idx = torch.unique(node_to_comm, return_inverse=True)
        num_comms = comm_ids.size(0)

        degrees = self._get_degrees(edges, weights, num_nodes)

        comm_degrees = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        comm_degrees.scatter_add_(0, node_to_comm_idx, degrees)

        edge_comm_s = node_to_comm_idx[edges[:, 0]]
        edge_comm_t = node_to_comm_idx[edges[:, 1]]

        internal_edges_mask = edge_comm_s == edge_comm_t
        internal_edge_weights = weights[internal_edges_mask]
        internal_edge_comms = edge_comm_s[internal_edges_mask]

        in_comm_weights = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        if internal_edge_weights.numel() > 0:
            in_comm_weights.scatter_add_(0, internal_edge_comms, internal_edge_weights)

        volumes = comm_degrees
        cuts = volumes - 2 * in_comm_weights

        comm_node_se = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        valid_comms_mask = volumes > 0
        if torch.sum(valid_comms_mask) > 0:
            comm_vol_nonzero = volumes[valid_comms_mask]
            # SE term: -(cut / Vol(G)) * log2(Vol(comm) / Vol(G))
            comm_node_se_valid = - (cuts[valid_comms_mask] / graph_vol) * torch.log2(comm_vol_nonzero / graph_vol)
            comm_node_se[valid_comms_mask] = comm_node_se_valid

        leaf_node_se = torch.zeros(num_comms, dtype=weights.dtype, device=device)
        comm_vol_by_node = volumes[node_to_comm_idx]
        valid_nodes_mask = (degrees > 0) & (comm_vol_by_node > 0)
        if torch.sum(valid_nodes_mask) > 0:
            # Leaf SE term: -(deg(i) / Vol(G)) * log2(deg(i) / Vol(comm(i)))
            leaf_node_se_contribs = - (degrees[valid_nodes_mask] / graph_vol) * torch.log2(
                degrees[valid_nodes_mask] / comm_vol_by_node[valid_nodes_mask])
            leaf_node_se.scatter_add_(0, node_to_comm_idx[valid_nodes_mask], leaf_node_se_contribs)

        return comm_ids, volumes, cuts, comm_node_se, leaf_node_se

    def _calc_delta_se_recursive(self, edges, weights, node_to_comm, num_nodes, graph_vol):
        """
        Calculates the change in structural entropy if two communities are merged.
        This is a core component of the greedy agglomerative clustering algorithm.
        """
        comm_ids, volumes, cuts, comm_node_se, leaf_node_se \
            = self._get_community_properties(edges, weights, node_to_comm, num_nodes, graph_vol)
        num_comms = comm_ids.size(0)
        if num_comms <= 1:
            return None, None

        comm_pairs = torch.combinations(torch.arange(num_comms, device=device), 2)
        idx1, idx2 = comm_pairs[:, 0], comm_pairs[:, 1]

        v1_vols, v2_vols = volumes[idx1], volumes[idx2]
        v1_cuts, v2_cuts = cuts[idx1], cuts[idx2]

        v1_comm_se, v2_comm_se = comm_node_se[idx1], comm_node_se[idx2]
        v1_leaf_se, v2_leaf_se = leaf_node_se[idx1], leaf_node_se[idx2]

        node_to_comm_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        comm_id_map = {id.item(): i for i, id in enumerate(comm_ids)}
        node_to_comm_idx = torch.tensor([comm_id_map[c.item()] for c in node_to_comm], device=device)

        edge_comm_s = node_to_comm_idx[edges[:, 0]]
        edge_comm_t = node_to_comm_idx[edges[:, 1]]

        comm_adj_matrix = torch.zeros((num_comms, num_comms), dtype=weights.dtype, device=device)
        comm_adj_matrix.index_put_((edge_comm_s, edge_comm_t), weights, accumulate=True)
        weights_between_comms = comm_adj_matrix[idx1, idx2] + comm_adj_matrix[idx2, idx1]

        original_se = v1_comm_se + v1_leaf_se + v2_comm_se + v2_leaf_se
        merged_comm_vols = v1_vols + v2_vols
        merged_comm_cuts = v1_cuts + v2_cuts - 2 * weights_between_comms

        vol_tensor = torch.tensor(graph_vol, device=device)
        merged_comm_vols_valid = merged_comm_vols > 0
        merged_comm_se_term = torch.zeros_like(merged_comm_vols)
        merged_comm_se_term[merged_comm_vols_valid] = - (
                merged_comm_cuts[merged_comm_vols_valid] / vol_tensor) * torch.log2(
            merged_comm_vols[merged_comm_vols_valid] / vol_tensor)

        merged_node_se_term = v1_leaf_se + v2_leaf_se

        v1_vols_valid = v1_vols > 0
        merged_node_se_term[v1_vols_valid] -= (v1_vols[v1_vols_valid] / vol_tensor) * torch.log2(
            v1_vols[v1_vols_valid] / merged_comm_vols[v1_vols_valid])

        v2_vols_valid = v2_vols > 0
        merged_node_se_term[v2_vols_valid] -= (v2_vols[v2_vols_valid] / vol_tensor) * torch.log2(
            v2_vols[v2_vols_valid] / merged_comm_vols[v2_vols_valid])

        merged_se = merged_comm_se_term + merged_node_se_term
        delta_SEs = merged_se - original_se
        return delta_SEs, comm_pairs

    def _build_node_to_community_map(self):
        """
        Builds a map from each original graph node ID to its corresponding
        leaf community node object in the tree.
        """
        self.graph_node_to_leaf_map = {}
        nodes_to_visit = [self.community_tree]
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if not current_node.children:
                if len(current_node.node_ids) == 1:
                    node_id = current_node.node_ids[0]
                    self.graph_node_to_leaf_map[node_id] = current_node
            else:
                for child in current_node.children:
                    nodes_to_visit.append(child)

    def calc_se_from_tree(self):
        if self.community_tree is None:
            print("Please run find_k_dim_entropy_tree() first.")
            return 0.0

        total_se = 0.0
        nodes_to_visit = [(self.community_tree, None)]

        while nodes_to_visit:
            current_node, parent_node = nodes_to_visit.pop(0)

            if parent_node is not None:
                if current_node.volume > 0 and parent_node.volume > 0:
                    se_term = -(current_node.cut / self.vol) * math.log2(current_node.volume / parent_node.volume)
                    total_se += se_term

            for child in current_node.children:
                nodes_to_visit.append((child, current_node))
        return total_se

    def calc_node_se_from_tree(self):
        if self.community_tree is None:
            print("Please run find_k_dim_entropy_tree() first.")
            return

        node_se_dict = {i: 0.0 for i in range(self.num_nodes)}
        nodes_to_visit = [(self.community_tree, None)]

        while nodes_to_visit:
            current_node, parent_node = nodes_to_visit.pop(0)
            if parent_node is not None:
                if current_node.volume > 0 and parent_node.volume > 0:
                    se_term = -(current_node.cut / self.vol) * math.log2(current_node.volume / parent_node.volume)
                    for node_id in current_node.node_ids:
                        node_se_dict[node_id] += se_term

            for child in current_node.children:
                nodes_to_visit.append((child, current_node))

        self.node_se = torch.tensor([node_se_dict[i] for i in range(self.num_nodes)], device=device)
        return self.node_se

    def _get_se_of_node_set(self, node_ids: list[int]) -> float:
        if self.node_se is None:
            self.calc_node_se_from_tree()

        se_sum = 0.0
        for node_id in node_ids:
            if node_id < self.num_nodes:
                se_sum += self.node_se[node_id].item()
        return se_sum

    def calc_conditional_se(self, src_ids: list[int], dst_ids: list[int]) -> float:
        """
        Calculates the conditional structural entropy H(dst_ids | src_ids)
        by strictly following the theoretical definition.
        H(Y|X) = sum(se_term(delta)) for all UNIQE delta in Y's branches that are NOT in X's branches.
        (Fix: Ensures unique community terms are summed)
        """
        if self.community_tree is None:
            print("Error: Community tree not built.")
            return 0.0

        # 1. 找出所有被 X 覆盖的社区 (Excluded Communities)
        excluded_communities = set()
        for src_id in src_ids:
            if src_id in self.graph_node_to_leaf_map:
                node = self.graph_node_to_leaf_map[src_id]
                while node is not None:
                    excluded_communities.add(node.ID)
                    node = node.parent

        # 2. 收集所有需要累加的唯一社区节点
        communities_to_sum = {}  # Key: CommunityNode.ID, Value: CommunityNode object

        for dst_id in dst_ids:
            if dst_id not in self.graph_node_to_leaf_map:
                continue

            current_node = self.graph_node_to_leaf_map[dst_id]

            while current_node is not None and current_node.parent is not None:
                # 检查是否被 X 覆盖
                if current_node.ID not in excluded_communities:
                    # 记录需要计算的社区项，确保唯一性
                    communities_to_sum[current_node.ID] = current_node

                # 无论是否累加，都必须继续向上追溯父节点
                current_node = current_node.parent

        # 3. 对收集到的唯一社区项进行求和
        total_conditional_se = 0.0

        for current_node in communities_to_sum.values():
            parent_node = current_node.parent  # 此时 parent_node 必然存在

            if current_node.volume > 0 and parent_node.volume > 0:
                se_term = -(current_node.cut / self.vol) * math.log2(
                    current_node.volume / parent_node.volume)
                total_conditional_se += se_term

        return total_conditional_se

    def calc_chain_rule_se(self, node_sequence: list[int]) -> float:
        if self.community_tree is None:
            print("Error: Community tree not built.")
            return 0.0
        if not node_sequence:
            return 0.0

        total_se = 0.0
        total_se += self.calc_conditional_se([], [node_sequence[0]])

        for i in range(1, len(node_sequence)):
            src_nodes = node_sequence[:i]
            dst_nodes = [node_sequence[i]]
            cond_se = self.calc_conditional_se(src_nodes, dst_nodes)
            total_se += cond_se

        return total_se

    def find_max_conditional_entropy_set(self, x_nodes: list[int], set_size: int) -> tuple[float, list]:
        if self.community_tree is None:
            print("Error: Community tree not built.")
            return 0.0, []

        available_nodes = [n for n in range(self.num_nodes) if n not in x_nodes]
        y_nodes = []

        for _ in tqdm(range(set_size), desc="Finding max conditional entropy set"):
            best_node = None
            max_delta_se = -1.0

            for candidate_node in available_nodes:
                current_y_plus_candidate = y_nodes + [candidate_node]
                delta_se = self.calc_conditional_se(x_nodes, current_y_plus_candidate)

                if delta_se > max_delta_se:
                    max_delta_se = delta_se
                    best_node = candidate_node

            if best_node is not None:
                y_nodes.append(best_node)
                available_nodes.remove(best_node)

        final_cond_se = self.calc_conditional_se(x_nodes, y_nodes)

        return final_cond_se, y_nodes

    def _find_common_ancestor_community_nodes(self, node_ids: list[int]) -> list[CommunityNode]:
        """
        Finds the highest-level community node that contains all given node_ids.
        Returns a list of such community nodes, ordered from top to bottom (root to leaf).
        """
        if not node_ids or not self.community_tree:
            return []

        # Start from the root and go down
        current_community = self.community_tree

        # Keep track of the path of community nodes that contain all x_set nodes
        common_ancestor_path = []

        nodes_to_check = set(node_ids)

        while True:
            # Check if all x_set nodes are in the current community
            if not nodes_to_check.issubset(set(current_community.node_ids)):
                # This community doesn't contain all nodes, so the previous one was the highest common ancestor
                break

            common_ancestor_path.append(current_community)

            found_next_level = False
            for child in current_community.children:
                if nodes_to_check.issubset(set(child.node_ids)):
                    current_community = child
                    found_next_level = True
                    break

            if not found_next_level:
                break

        return common_ancestor_path

    def find_max_chain_conditional_entropy_set(self, x_set: list[int], set_size: int, beam_width: int = 1) \
            -> list[tuple[float, list]]:
        """
        Finds a set of nodes (y_set) that maximizes the chain conditional structural entropy
        H(Y|X) using a beam search approach. (Legacy: No set-deduplication)
        """
        if self.community_tree is None:
            print("Error: Community tree not built.")
            return []

        # 1. 找到 x_set 中所有节点的共同祖先社区
        common_ancestor_nodes = self._find_common_ancestor_community_nodes(x_set)

        if not common_ancestor_nodes:
            print("Warning: No common ancestor community found for the given x_set. Using all nodes.")
            search_space_nodes = [n for n in range(self.num_nodes)]
        else:
            # 2. 将搜索空间缩小到这个共同祖先社区的成员节点
            search_community = common_ancestor_nodes[0]
            search_space_nodes = search_community.node_ids

        # 初始化 beam，包含一个空的 y_set 及其初始条件熵（0）
        # initial_se_val = self.calc_chain_rule_se(x_set) # Not needed for pure gain tracking
        beam = [(0.0, [])]  # (total_conditional_se_gain, y_set)

        with tqdm(total=set_size, desc="Finding max chain conditional entropy set (Beam Search)") as pbar:
            for _ in range(set_size):
                new_beam_candidates = []

                # 对 beam 中的每个当前最佳序列进行扩展
                for current_se_gain, current_y_set in beam:
                    current_chain = list(x_set) + current_y_set

                    # 可用的候选节点是在搜索空间内且未被使用的节点
                    available_nodes = [n for n in search_space_nodes if n not in current_chain]

                    for candidate_node in available_nodes:
                        # 计算新增一个节点后的总条件熵增益
                        # 新增的熵增益 = H(candidate | current_chain)
                        delta_se = self.calc_conditional_se(current_chain, [candidate_node])

                        new_se_gain = current_se_gain + delta_se
                        new_y_set = current_y_set + [candidate_node]

                        new_beam_candidates.append((new_se_gain, new_y_set))

                # 如果没有新的候选节点，则终止
                if not new_beam_candidates:
                    break

                # 对所有新的候选序列进行排序，并选择前 k 个作为新的 beam
                new_beam_candidates.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam_candidates[:beam_width]

                pbar.update(1)

        # 计算最终的 H(Y|X) 并准备返回结果
        final_results = []
        # h_x = self.calc_chain_rule_se(x_set) # Not strictly needed if total_se_gain is correct
        for total_se_gain, y_set in beam:
            final_cond_se = total_se_gain  # total_se_gain is the accumulated chain conditional entropy
            final_results.append((final_cond_se, y_set))

        return final_results

    def find_all_chain_conditional_entropy_sets(self, x_set: list[int], set_size: int) \
            -> list[tuple[float, list]]:
        """
        找到所有长度为 set_size 的节点序列 Y，计算链式条件结构熵 H(Y|X)，
        并按熵值从大到小排序。
        """
        if self.community_tree is None:
            print("Error: Community tree not built.")
            return []
        if set_size <= 0:
            return [(0.0, [])]

        # 确定搜索空间：所有未被 x_set 包含的节点
        candidate_nodes = [n for n in range(self.num_nodes) if n not in x_set]

        # 检查候选节点数量是否足够
        if len(candidate_nodes) < set_size:
            print(
                f"Warning: Only {len(candidate_nodes)} nodes available, requested set size is {set_size}. Cannot generate all permutations.")
            set_size = len(candidate_nodes)
            if set_size == 0:
                return [(0.0, [])]

        # 1. 生成所有可能的 Y 集合的序列（全排列）
        all_y_permutations = list(itertools.permutations(candidate_nodes, set_size))

        print(f"Total number of sequences to evaluate: {len(all_y_permutations)}")

        results = []

        # 预先计算 H(X)
        h_x = self.calc_chain_rule_se(x_set)

        # 2. 遍历每个序列，计算 H(Y|X)
        for y_set_sequence in tqdm(all_y_permutations, desc="Evaluating all sequences"):
            y_set_sequence = list(y_set_sequence)

            # 完整序列 Z = X + Y
            full_sequence = x_set + y_set_sequence

            # 计算 H(X, Y) = H(Z)
            h_combined = self.calc_chain_rule_se(full_sequence)

            # 计算 H(Y|X) = H(X, Y) - H(X)
            conditional_se = h_combined - h_x

            results.append((conditional_se, y_set_sequence))

        # 3. 按条件熵 H(Y|X) 降序排序
        results.sort(key=lambda x: x[0], reverse=True)

        return results

    def find_max_global_conditional_entropy_sets(self, x_set: list[int], set_size: int, beam_width: int = 1) \
            -> list[tuple[float, list]]:
        """
        Finds a set of nodes (y_set) that maximizes the chain conditional structural entropy H(Y|X)
        using a beam search approach, ensuring that the resulting node SETS Y are unique at each step,
        and prioritizing the sequence that yields the maximum conditional entropy for that unique set.
        (NEW: Implements set-deduplication within the beam search loop)
        """

        if self.community_tree is None:
            print("Error: Community tree not built.")
            return []

        common_ancestor_nodes = self._find_common_ancestor_community_nodes(x_set)

        if not common_ancestor_nodes:
            search_space_nodes = [n for n in range(self.num_nodes)]
        else:
            search_community = common_ancestor_nodes[0]
            search_space_nodes = search_community.node_ids

        # 初始化 beam，包含一个空的 y_set 及其初始条件熵（0）
        # beam 结构: (total_conditional_se_gain, y_set)
        beam = [(0.0, [])]

        with tqdm(total=set_size, desc="Finding max unique conditional entropy sets (Beam Search)") as pbar:
            for _ in range(set_size):
                new_beam_candidates = []

                # 对 beam 中的每个当前最佳序列进行扩展
                for current_se_gain, current_y_set in beam:
                    current_chain = list(x_set) + current_y_set

                    # 可用的候选节点是在搜索空间内且未被使用的节点
                    available_nodes = [n for n in search_space_nodes if n not in current_chain]

                    for candidate_node in available_nodes:
                        # 计算新增一个节点后的总条件熵增益
                        # 新增的熵增益 = H(candidate | current_chain)
                        delta_se = self.calc_conditional_se(current_chain, [candidate_node])

                        new_se_gain = current_se_gain + delta_se
                        new_y_set = current_y_set + [candidate_node]

                        new_beam_candidates.append((new_se_gain, new_y_set))

                # 如果没有新的候选节点，则终止
                if not new_beam_candidates:
                    break

                # --- 关键修改：去重和选择最佳集合 ---

                # 1. 创建字典用于存储每个集合的最佳增益序列
                # 键: 排序后的集合元组，值: (total_se_gain, y_set_sequence)
                unique_candidates = {}

                # 2. 遍历所有候选序列，进行去重和择优
                for total_se_gain, new_y_set in new_beam_candidates:
                    # 规范化集合表示 (排序后的元组)，这是去重的基础
                    # 只有 Y 集合中的节点参与排序，X 不参与
                    set_key = tuple(sorted(new_y_set))

                    # 检查: 如果该集合是第一次出现，或新序列的条件熵增益更高，则更新
                    if set_key not in unique_candidates or total_se_gain > unique_candidates[set_key][0]:
                        unique_candidates[set_key] = (total_se_gain, new_y_set)

                # 3. 将唯一的最佳集合序列转换为列表
                unique_results = list(unique_candidates.values())

                # 4. 按条件熵增益从大到小排序
                unique_results.sort(key=lambda x: x[0], reverse=True)

                # 5. 选择前 beam_width 个作为新的 beam
                beam = unique_results[:beam_width]

                pbar.update(1)

        # 计算最终的 H(Y|X) 并准备返回结果
        final_results = []
        # total_se_gain is the accumulated chain conditional entropy H(Y|X)
        for final_cond_se, y_set in beam:
            # y_set 是产生最大条件熵的那个序列，即使集合相同，我们保留这个最优序列
            final_results.append((final_cond_se, y_set))

        return final_results

    def find_max_local_chain_conditional_entropy_sets(self, x_set: list[int], set_size: int, beam_width: int = 1) \
            -> list[tuple[float, list]]:
        """
        Finds a set of nodes (y_set) that maximizes the chain conditional structural entropy H(Y|X)
        using a Beam Search with a dynamic, locally-focused search space.

        The search space expands bottom-up: starting from the closest common ancestor community
        of X, and iteratively moving to the parent community at each step.

        Args:
            x_set (list[int]): The initial set of nodes to condition on.
            set_size (int): The number of nodes to find for the set Y.
            beam_width (int): The number of top unique candidate sets to keep at each step.

        Returns:
            list[tuple[float, list]]: A list of tuples (total_cond_se, y_set) for the top beam_width unique sets found.
        """

        if self.community_tree is None:
            print("Error: Community tree not built.")
            return []

        # 1. 确定初始的社区层级路径 (从最深共同祖先到根节点)
        # common_ancestor_path: [C_deepest, C_parent, ..., C_root]
        common_ancestor_path = self._find_common_ancestor_community_nodes(x_set)

        if not common_ancestor_path:
            # 如果没有共同祖先（例如，X只包含一个节点，或树未完全构建），则使用整个图作为搜索空间。
            community_sequence_for_steps = [self.community_tree] * set_size
        else:
            # common_ancestor_path 已经是按深度降序排列的，从最深到根。
            # 我们需要这个路径的社区节点，用于逐步扩展。
            community_sequence_for_steps = common_ancestor_path

        # 填充社区序列，直到达到 set_size，如果需要，就重复根节点。
        while len(community_sequence_for_steps) < set_size:
            community_sequence_for_steps.append(self.community_tree)  # 根节点包含所有图节点

        # 2. 初始化 Beam Search
        # beam 结构: (total_conditional_se_gain, y_set)
        beam = [(0.0, [])]

        with tqdm(total=set_size, desc="Finding max local chain conditional entropy sets (Beam Search)") as pbar:
            for step_t in range(set_size):

                # A. 确定本步骤的搜索空间 (自底向上扩展)
                # 步骤 t 对应 community_sequence_for_steps[t]
                current_search_community = community_sequence_for_steps[step_t]

                # 搜索空间是当前社区的所有成员节点
                current_search_space_nodes = current_search_community.node_ids

                new_beam_candidates = []

                # B. 对 beam 中的每个当前最佳序列进行扩展
                for current_se_gain, current_y_set in beam:
                    current_chain = list(x_set) + current_y_set

                    # 可用的候选节点是在当前搜索空间内且未被使用的节点
                    available_nodes = [n for n in current_search_space_nodes if n not in current_chain]

                    for candidate_node in available_nodes:
                        # 计算新增一个节点后的总条件熵增益 H(candidate | current_chain)
                        delta_se = self.calc_conditional_se(current_chain, [candidate_node])

                        new_se_gain = current_se_gain + delta_se
                        new_y_set = current_y_set + [candidate_node]

                        new_beam_candidates.append((new_se_gain, new_y_set))

                # C. 去重和选择最佳集合 (沿用 unique Beam Search 的高效逻辑)
                if not new_beam_candidates:
                    break

                unique_candidates = {}
                for total_se_gain, new_y_set in new_beam_candidates:
                    set_key = tuple(sorted(new_y_set))

                    if set_key not in unique_candidates or total_se_gain > unique_candidates[set_key][0]:
                        unique_candidates[set_key] = (total_se_gain, new_y_set)

                unique_results = list(unique_candidates.values())
                unique_results.sort(key=lambda x: x[0], reverse=True)

                # 选择前 beam_width 个作为新的 beam
                beam = unique_results[:beam_width]

                pbar.update(1)

        # 3. 计算最终结果并返回
        final_results = []
        for final_cond_se, y_set in beam:
            final_results.append((final_cond_se, y_set))

        return final_results


    def find_k_dim_entropy_tree(self, k_dim: int = 2):
        if k_dim <= 1:
            self.community_tree = CommunityNode(list(range(self.num_nodes)))
            self.all_community_nodes[self.community_tree.ID] = self.community_tree
            self.graph_node_to_leaf_map = {node_id: self.community_tree for node_id in range(self.num_nodes)}
            return self.community_tree

        root_comm_node = CommunityNode(list(range(self.num_nodes)))
        self.community_tree = root_comm_node
        self.all_community_nodes[root_comm_node.ID] = root_comm_node
        current_level_comms = [root_comm_node]

        for current_level in range(1, k_dim + 1):
            next_level_comms = []
            print(f"Finding communities for dimension {current_level} (tree level {current_level})...")

            for parent_comm in current_level_comms:
                subgraph_nodes_orig_ids = torch.tensor(parent_comm.node_ids, device=device)
                if len(subgraph_nodes_orig_ids) <= 1:
                    child_comm_node = CommunityNode(parent_comm.node_ids, parent=parent_comm)
                    parent_comm.children.append(child_comm_node)
                    next_level_comms.append(child_comm_node)
                    self.all_community_nodes[child_comm_node.ID] = child_comm_node
                    if parent_comm.node_ids:
                        node_id = parent_comm.node_ids[0]
                        node_deg = self.degrees[node_id].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                node_map = {node_id.item(): i for i, node_id in enumerate(subgraph_nodes_orig_ids)}
                edges_mask = torch.isin(self.edges, subgraph_nodes_orig_ids).all(dim=1)
                sub_edges_orig_ids = self.edges[edges_mask]
                sub_weights = self.weights[edges_mask]

                if sub_edges_orig_ids.numel() == 0:
                    for node_id in subgraph_nodes_orig_ids:
                        child_comm_node = CommunityNode([node_id.item()], parent=parent_comm)
                        parent_comm.children.append(child_comm_node)
                        next_level_comms.append(child_comm_node)
                        self.all_community_nodes[child_comm_node.ID] = child_comm_node
                        node_deg = self.degrees[node_id.item()].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                sub_edges_relative_ids = torch.tensor(
                    [[node_map[u.item()], node_map[v.item()]] for u, v in sub_edges_orig_ids], dtype=torch.long,
                    device=device)
                sub_num_nodes = len(subgraph_nodes_orig_ids)
                sub_graph_vol = torch.sum(
                    self._get_degrees(sub_edges_relative_ids, sub_weights, sub_num_nodes)).item()
                if sub_graph_vol == 0:
                    for node_id in subgraph_nodes_orig_ids:
                        child_comm_node = CommunityNode([node_id.item()], parent=parent_comm)
                        parent_comm.children.append(child_comm_node)
                        next_level_comms.append(child_comm_node)
                        self.all_community_nodes[child_comm_node.ID] = child_comm_node
                        node_deg = self.degrees[node_id.item()].item()
                        child_comm_node.volume = node_deg
                        child_comm_node.cut = node_deg
                    continue

                sub_node_to_comm = torch.arange(sub_num_nodes, device=device)

                while True:
                    sub_comm_ids, _, _, _, _ = self._get_community_properties(
                        sub_edges_relative_ids, sub_weights, sub_node_to_comm, sub_num_nodes, sub_graph_vol
                    )
                    delta_SEs, comm_pairs = self._calc_delta_se_recursive(
                        sub_edges_relative_ids, sub_weights, sub_node_to_comm, sub_num_nodes, sub_graph_vol
                    )
                    if delta_SEs is None:
                        break
                    min_delta_SE, min_idx = torch.min(delta_SEs, dim=0)
                    if min_delta_SE < 0:
                        best_comm_idx1, best_comm_idx2 = comm_pairs[min_idx]
                        best_comm1 = sub_comm_ids[best_comm_idx1].item()
                        best_comm2 = sub_comm_ids[best_comm_idx2].item()
                        mask_comm2 = sub_node_to_comm == best_comm2
                        sub_node_to_comm[mask_comm2] = best_comm1
                    else:
                        break

                sub_comm_ids_unique, _ = torch.unique(sub_node_to_comm, return_inverse=True)
                for sub_comm_id in sub_comm_ids_unique:
                    sub_comm_id = sub_comm_id.item()
                    member_relative_ids = (sub_node_to_comm == sub_comm_id).nonzero(
                        as_tuple=False).squeeze().cpu().tolist()
                    if not isinstance(member_relative_ids, list):
                        member_relative_ids = [member_relative_ids]

                    member_orig_ids = [subgraph_nodes_orig_ids[i].item() for i in member_relative_ids]
                    child_comm_node = CommunityNode(member_orig_ids, parent=parent_comm)
                    parent_comm.children.append(child_comm_node)
                    next_level_comms.append(child_comm_node)
                    self.all_community_nodes[child_comm_node.ID] = child_comm_node

                    subgraph_nodes_orig_ids_tensor = torch.tensor(member_orig_ids, device=device)
                    child_comm_node.volume = torch.sum(self.degrees[subgraph_nodes_orig_ids_tensor]).item()

                    edges_mask = torch.isin(self.edges, subgraph_nodes_orig_ids_tensor).all(dim=1)
                    internal_sum = torch.sum(self.weights[edges_mask]).item()
                    child_comm_node.cut = child_comm_node.volume - 2 * internal_sum

            if not next_level_comms:
                break
            current_level_comms = next_level_comms

        self._build_node_to_community_map()

        return self.community_tree

    def _get_networkx_graph(self):
        edges_cpu = self.edges.cpu().tolist()
        weights_cpu = self.weights.cpu().tolist()
        relations_cpu = self.relations
        G = nx.DiGraph()
        for i, ((u, v), w) in enumerate(zip(edges_cpu, weights_cpu)):
            relation = relations_cpu[i] if relations_cpu and i < len(relations_cpu) else None
            G.add_edge(u, v, weight=w, relation=relation)
        if hasattr(self, 'node_se') and self.node_se is not None:
            node_se_cpu = self.node_se.cpu().tolist()
            for i, se in enumerate(node_se_cpu):
                if i in G.nodes:
                    G.nodes[i]['se'] = se
        return G

    def find_top_k_se_paths(self, s_node: any, path_length: int, k: int = 5) -> list[tuple[float, list]]:
        if not hasattr(self, 'node_se') or self.node_se is None:
            print("Calculating node SE from the tree first...")
            self.calc_node_se_from_tree()

        G = self._get_networkx_graph()
        if s_node not in G:
            print(f"Error: Starting node '{s_node}' not found in the graph.")
            return []

        all_paths_with_se = []

        def dfs(current_node, current_path_repr, current_se_sum, visited_nodes):
            current_edges_count = (len(current_path_repr) - 1) // 2
            if current_edges_count == path_length:
                all_paths_with_se.append((current_se_sum, list(current_path_repr)))
                return
            if current_edges_count > path_length:
                return
            for _, v, edge_data in G.edges(current_node, data=True):
                neighbor = v
                if neighbor not in visited_nodes:
                    relation = edge_data.get('relation', 'unknown_relation')
                    neighbor_se = G.nodes[neighbor].get('se', 0.0)
                    dfs(neighbor, current_path_repr + [relation, neighbor], current_se_sum + neighbor_se,
                        visited_nodes | {neighbor})

        initial_se = G.nodes[s_node].get('se', 0.0)
        dfs(s_node, [s_node], initial_se, {s_node})
        all_paths_with_se.sort(key=lambda x: x[0], reverse=True)
        return all_paths_with_se[:k]

    def calc_first_order_se(self) -> float:
        """
        Calculates the total first-order entropy H^1(G) of the graph G.
        H^1(G) = -sum_{i in V} p_i log_2 p_i, where p_i = deg(i) / (2m).
        """

        # Calculate p_i (normalized degrees)
        p_i = self.degrees / self.vol

        # Filter out nodes with zero degree (p_i=0) to avoid log(0)
        valid_mask = p_i > 0
        p_i_valid = p_i[valid_mask]

        # Calculate H^1(G) = -sum(p_i * log2(p_i))
        h1_G = -torch.sum(p_i_valid * torch.log2(p_i_valid))

        return h1_G.item()

    def calc_sweet_spot_metric(self) -> float:
        """
        Calculates the Sweet Spot Metric, theta^T(G), which quantifies the
        reduction in information achieved by the hierarchical structure.
        theta^T(G) = 1 - (H^T(G) / H^1(G)).
        """
        h_T_G = self.calc_se_from_tree()
        h_1_G = self.calc_first_order_se()

        theta_T_G = 1.0 - (h_T_G / h_1_G)

        return theta_T_G


if __name__ == "__main__":
    print(f"Using device: {device}")

    edges_cpu = torch.tensor([
        [0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]
    ], dtype=torch.long)
    weights_cpu = torch.tensor([
        0.5, 0.7, 1.0, 0.43, 0.62, 0.92, 0.13, 0.05, 0.76, 0.61
    ], dtype=torch.float)
    relations_list = [
        "rel_a", "rel_b", "rel_c", "rel_d", "rel_e", "rel_f", "rel_g", "rel_h", "rel_i", "rel_j"
    ]

    seg = StructEntropy(edges=edges_cpu, weights=weights_cpu, relations=relations_list)

    k_dim_to_find = 3
    print(f"\n--- Finding Minimum Entropy Tree with K-Dimension = {k_dim_to_find} ---")
    community_tree_3d = seg.find_k_dim_entropy_tree(k_dim=k_dim_to_find)


    def print_tree(node, level=0):
        indent = "  " * level
        print(f"{indent}Community (Level {level}): {node.node_ids}")
        for child in node.children:
            print_tree(child, level + 1)


    print("\n--- Final 3D Community Tree ---")
    print_tree(community_tree_3d)

    print("\n--- Calculating Final SE from the Tree ---")
    se_value_3d = seg.calc_se_from_tree()
    print(f"Final SE from tree: {se_value_3d:.4f}")

    seg.calc_node_se_from_tree()

    print("\n--- Calculating Conditional SE (Multiple Nodes) ---")
    src_nodes_list = [0, 2]
    dst_nodes_list = [1, 3]
    cond_se_multi = seg.calc_conditional_se(src_nodes_list, dst_nodes_list)
    print(f"Conditional SE from nodes {src_nodes_list} to nodes {dst_nodes_list}: {cond_se_multi:.4f}")

    print("\n--- Finding Max Conditional Entropy Set ---")
    x_set = [0, 2]
    set_size = 2
    max_se, y_set = seg.find_max_conditional_entropy_set(x_set, set_size)
    print(f"Given condition set X={x_set}, max conditional SE is {max_se:.4f} for set Y* = {y_set}")

    print("\n--- Calculating Chain Rule SE ---")
    node_sequence = [0, 2, 1, 3]
    chain_se = seg.calc_chain_rule_se(node_sequence)
    print(f"Chain rule SE for sequence {node_sequence}: {chain_se:.4f}")

    # Using the legacy beam search (no set deduplication)
    max_se, y_set = seg.find_max_chain_conditional_entropy_set(x_set, set_size)[0]
    print(
        f"Given Chain condition SE set X={x_set}, max Chain conditional SE (Legacy) is {max_se:.4f} for set Y* = {y_set}")

    # Using the new, unique-set beam search
    top_unique_sets = seg.find_max_unique_conditional_entropy_sets(x_set, set_size, beam_width=3)
    print(f"\n--- Finding Max UNIQUE Conditional Entropy Sets (Beam Search) ---")
    for rank, (max_se, y_set_seq) in enumerate(top_unique_sets):
        print(f"Rank {rank + 1}: Max UNIQUE Chain cond. SE is {max_se:.4f} for sequence Y* = {y_set_seq}")

    print("\n--- Finding Paths based on Node SE from the Tree ---")
    s_node = 0
    path_length = 2
    top_paths_3d = seg.find_top_k_se_paths(s_node, path_length, k=3)

    print(f"Top 3 paths of length {path_length} starting from node {s_node}:")
    if top_paths_3d:
        for se_sum, path_repr in top_paths_3d:
            print(f"  SE Sum: {se_sum:.4f}, Path: {path_repr}")
    else:
        print("No paths found.")