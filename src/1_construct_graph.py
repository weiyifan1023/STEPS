import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import pickle
import random
import glob, os
import pandas as pd
from typing import List, Dict, Any, Set, Union
from tqdm import tqdm
import pdb
import ast




def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat


def save_jsonl(sample_ls, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for ipt in sample_ls:
            json_str = json.dumps(ipt, ensure_ascii=False)
            f.write(json_str + '\n')


def load_parquet(
        file_path="/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/infinityinstruct_Gen"):
    """
    加载 Parquet 文件，返回合并后的 DataFrame。
    """
    parquet_files = glob.glob(os.path.join(file_path, "*.parquet"))
    all_dfs = []

    if not parquet_files:
        print(f"Error: No Parquet files found in {file_path}")
        return pd.DataFrame()

    for file in tqdm(parquet_files, desc="Loading Parquet files"):
        try:
            df = pd.read_parquet(file, columns=['id', 'label'])
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read file {file}. Error: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully loaded and combined {len(combined_df)} raw samples from {len(parquet_files)} files.")
        return combined_df
    else:
        return pd.DataFrame()



def build_graph(df: pd.DataFrame):
    """
    从 Pandas DataFrame 中提取 'label' 字典内的 'ability_zh' 列表，并构建共现图。
    加入了对字符串格式字典和 NumPy 数组的解析。
    """
    if 'label' not in df.columns:
        print("Error: DataFrame must contain a 'label' column.")
        return nx.Graph()

    def extract_zh_abilities(label_data):
        label_dict = None
        if isinstance(label_data, str):
            try:
                label_dict = ast.literal_eval(label_data)
            except (ValueError, SyntaxError):

                return None
        elif isinstance(label_data, dict):
            label_dict = label_data
        else:
            return None

        if isinstance(label_dict, dict):
            abilities = label_dict.get('ability_zh', [])

            if isinstance(abilities, np.ndarray):
                try:
                    abilities = abilities.tolist()
                except AttributeError:
                    pass

            if isinstance(abilities, list) and len(abilities) > 0:
                return abilities

        return None


    print("1. Extracting 'ability_zh' tags and counting occurrences...")

    zh_abilities = df['label'].apply(extract_zh_abilities).dropna()

    if zh_abilities.empty:
        print("Error: No valid 'ability_zh' tags found after extraction and filtering.")
        return nx.Graph()  # 退出

    tag_ls = []
    for label_list in tqdm(zh_abilities, desc="Collecting all tags"):
        tag_ls.extend(label_list)

    tag_counts = Counter(tag_ls)
    print(f"Found {len(tag_counts)} distinct tags.")

    edge_idx_dict = {}

    print("2. Counting tag co-occurrences...")

    for tag_ls_tmp in tqdm(zh_abilities, desc="Calculating co-occurrences"):
        for i in range(len(tag_ls_tmp)):
            tag_i = tag_ls_tmp[i]
            for j in range(i + 1, len(tag_ls_tmp)):
                tag_j = tag_ls_tmp[j]

                edge_idx_dict.setdefault(tag_i, {})
                edge_idx_dict.setdefault(tag_j, {})

                edge_idx_dict[tag_i][tag_j] = edge_idx_dict[tag_i].get(tag_j, 0) + 1
                edge_idx_dict[tag_j][tag_i] = edge_idx_dict[tag_j].get(tag_i, 0) + 1

    print('3. Calculating edge weights (PMI normalized)...')
    for k_1 in edge_idx_dict.keys():
        for k_2 in edge_idx_dict[k_1].keys():
            co_occurrence = edge_idx_dict[k_1][k_2]
            count_k1 = tag_counts.get(k_1, 0)
            count_k2 = tag_counts.get(k_2, 0)

            edge_idx_dict[k_1][k_2] = (float(co_occurrence) + 2) / (count_k1 + 1) / (count_k2 + 1)

    print('4. Building graph...')
    G = nx.Graph()
    for k_1 in tqdm(edge_idx_dict.keys(), desc="Adding edges to graph"):
        for k_2, weight in edge_idx_dict[k_1].items():
            if k_1 != k_2:
                G.add_edge(k_1, k_2, weight=weight)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G



if __name__ == '__main__':
    parquet_data_path = '/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/infinityinstruct_Gen'

    random.seed(2024)

    df_data = load_parquet(parquet_data_path)

    if df_data.empty:
        print("Data loading failed. Exiting.")
    else:
        dat_sample = df_data

        graph = build_graph(dat_sample)

        output_file = '/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/skill_graph_zh_ability.pkl'
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Graph saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving graph: {e}")