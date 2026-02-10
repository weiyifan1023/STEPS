import pickle
import json
import os
import random
import time
import ast
import glob
import gc
import argparse
import re
import pandas as pd
from typing import List, Dict, Any, Set, Union
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=4, help="k value for file paths")
parser.add_argument("--gen_num", type=int, default=2000, help="number of generated data")
args = parser.parse_args()

base_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/k={args.k}"
final_output_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/generated_complex_data_k={args.k}_EN.jsonl"
tag_path = "/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/infinityinstruct_Gen"
atomic_data_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/atomic_data_infinity_EN.jsonl"


def gen_response(messages, max_retries: int = 3, initial_sleep: int = 1):
    client = OpenAI(
        api_key="",
        base_url="")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )

            content = response.choices[0].message.content
            return content

        except Exception as e:
            print(f"API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_sleep * (2 ** attempt)
                print(f"Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"API request failed after {max_retries} attempts. Skipping this sample.")
                return None


def string_to_builtin_type(data_string: str) -> Union[Any, str]:
    """
    尝试将一个字符串安全地转换为 Python 的内置类型。
    """
    stripped_string = data_string.strip()
    if not stripped_string:
        return data_string
    try:
        return ast.literal_eval(stripped_string)
    except (ValueError, SyntaxError):
        return data_string


system_prompt = r"""You are the Synergistic Content Architect, an Advanced Universal Content Creation AI specializing in Constraint Synthesis, Multi-Domain Fusion, and the generation of Composite Skill Narratives.
Your Core Mission is to execute a Composite Generalization Task:
1.  Analyze & Map (Atomic Skills): Study all independent text fragments and their associated Atomic Skill Tags ($\text{S}_{\text{A}1}, \text{S}_{\text{A}2}, \dots$) provided by the user. Comprehend the core knowledge and technical details embedded in each fragment.
2.  Mandatorily Incorporate (Composite Skill): Seamlessly integrate the foundational information from *all* fragments to create a single, novel, and highly complex output. This new content ($\text{Y}$) MUST fully utilize and demonstrate a Composite Skill Set ($\text{S}_{\text{C}} = \text{S}_{\text{A}1} \cup \text{S}_{\text{A}2} \cup \dots$). The output must serve as a functional example or application that intrinsically relies on this new, combined capability.
3.  Logical Coherence: If the sources cover disparate fields (e.g., historical linguistics, quantum computing, mycology), establish a sophisticated, unifying project background (a 'meta-context') to logically fuse these domains into a single, plausible scenario or dialogue.
4.  Demonstrate Full Complexity: The resulting output must not merely be a concatenation of sources, but a deeper rewrite where the vocabulary, logical flow, and technical requirements of EVERY SINGLE ORIGINAL Atomic Skill Tag are interwoven into a coherent, complex whole. ENSURE FULL SKILL COVERAGE.

Strict Output Format Adherence:
* The final output must be a single Python list of dictionaries, representing a synthesized multi-turn conversation (Dialogue Fusion).
* Each dictionary must strictly follow the format: `{'role': 'user', 'content': '...'}` for instructions, and `{'role': 'assistant', 'content': '...'}` for responses.
* You can generate a single pair or multiple pairs in the list `[]`.

Constraint Directives:
* Treat the original source material (including all user/assistant turns, and skill tag info) as the foundational lexicon and logical structure for the new creation.
* Directly output the final formatted Python list. Do not include any explanations, introductory statements, or extra formatting tags (e.g., no markdown code block identifiers like ```python).
"""



file_paths = glob.glob(f"{base_path}/optimal_combinations_*.pkl")
file_paths.sort(key=lambda x: int(re.search(r'optimal_combinations_(\d+)\.pkl', x).group(1)))

loaded_data: List[Dict[str, Any]] = []
for file_path in file_paths:
    try:
        with open(file_path, 'rb') as f:
            file_data = pickle.load(f)
        loaded_data.extend(file_data)
        print(f"Successfully loaded {len(file_data)} combinations from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(f"Total loaded {len(loaded_data)} optimal combinations from {len(file_paths)} files.")

random.shuffle(loaded_data)



atomic_data: List[Dict[str, Any]] = []


if os.path.exists(atomic_data_path):
    print(f"\nExisting atomic_data file detected at {atomic_data_path}, loading...")
    try:
        with open(atomic_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    atomic_data.append(json.loads(line))
        print(f"Successfully loaded {len(atomic_data)} atomic data samples. (Assumed English)")
    except Exception as e:
        print(f"Error loading atomic_data file: {e}. Reprocessing raw data.")
        atomic_data = []
else:
    print(f"\nAtomic_data file not detected at {atomic_data_path}. Processing raw data...")

    data: List[Dict[str, Any]] = []

    parquet_files = glob.glob(os.path.join(tag_path, "*.parquet"))

    if not parquet_files:
        print(f"Error: No .parquet files found in directory {tag_path}")
        data = []
    else:
        all_dfs = []
        try:
            for file in parquet_files:
                df = pd.read_parquet(file, columns=['id', 'conversations', 'label', 'langdetect'])
                all_dfs.append(df)

            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Successfully loaded and combined {len(combined_df)} raw samples from {len(parquet_files)} files.")


            pre_filter_count = len(combined_df)
            combined_df = combined_df[combined_df['langdetect'] == 'en'].copy()
            print(
                f"Filtering complete. Retained {len(combined_df)} samples with 'langdetect': 'en' (Dropped {pre_filter_count - len(combined_df)} non-English samples).")

            for index, row in combined_df.iterrows():
                try:
                    conversation_list = []
                    convs = row['conversations']
                    if hasattr(convs, '__iter__'):
                        for conv in convs:
                            conversation_list.append({
                                'role': 'user' if conv.get('from') == 'human' else 'assistant',
                                'content': conv.get('value', '')
                            })

                    abilities = row['label'].get('ability_zh', [])

                    if isinstance(abilities, np.ndarray):
                        abilities = abilities.tolist()
                    elif not isinstance(abilities, list):
                        abilities = [str(abilities)] if abilities else []

                    if not abilities:
                        continue

                    new_sample = {
                        'id': row['id'],
                        'label': {'ability': abilities},
                        'content': conversation_list,
                        'langdetect': row.get('langdetect', 'unknown'),  # 此时必为 'en'
                    }
                    data.append(new_sample)

                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue

            print(f"Successfully extracted and formatted {len(data)} English raw samples.")
        except Exception as e:
            print(f"Error loading or processing Parquet files: {e}")
            data = []

    if data:
        tag_counts = Counter()

        with open(atomic_data_path, 'w', encoding='utf-8') as outfile:
            for sample in data:
                abilities = sample.get('label', {}).get('ability', [])

                if abilities:
                    # ----------------- 标签选择逻辑 (保持不变) -----------------
                    ability_counts = {tag: tag_counts[tag] for tag in abilities}
                    min_count = min(ability_counts.values()) + random.randint(15, 25)

                    least_frequent_abilities = [
                        tag for tag, count in ability_counts.items()
                        if count <= min_count
                    ]

                    selected_ability = random.choice(least_frequent_abilities)

                    tag_counts[selected_ability] += 1

                    new_sample = sample.copy()
                    new_sample['label'] = {'ability': [selected_ability]}
                    new_sample['single_ability_tag'] = selected_ability

                    atomic_data.append(new_sample)

                    outfile.write(json.dumps(new_sample, ensure_ascii=False) + '\n')

        print(f"Generated and saved {len(atomic_data)} atomic (single-tag, English) samples to {atomic_data_path}.")
        print("Atomic Ability Tag Distribution (Top 10):", tag_counts.most_common(10))

ability_to_samples: Dict[str, List[Dict[str, Any]]] = {}

for sample in atomic_data:
    tag = sample.get('single_ability_tag')
    if tag:
        if tag not in ability_to_samples:
            ability_to_samples[tag] = []
        ability_to_samples[tag].append(sample)

print(f"Created index for {len(ability_to_samples)} distinct English ability tags.")


llm_inputs: List[Dict[str, Any]] = []

for combo_info in loaded_data:
    optimal_combination: List[str] = [combo_info['start_ability']] + combo_info['optimal_combination']
    combined_content = []
    current_combo_sample_ids = []

    is_complete_combo = True
    for target_tag in optimal_combination:

        if target_tag in ability_to_samples and ability_to_samples[target_tag]:
            available_samples = [
                s for s in ability_to_samples[target_tag]
                if s['id'] not in current_combo_sample_ids
            ]

            if not available_samples:
                selected_sample = random.choice(ability_to_samples[target_tag])
            else:
                selected_sample = random.choice(available_samples)

            current_combo_sample_ids.append(selected_sample['id'])
            sample_content = selected_sample.get('content', [])

            combined_content.append(
                {"role": "skill_tag",
                 "content": f"The Skill involved in content is {target_tag}"}
            )
            combined_content.extend(sample_content)

        else:
            print(
                f"Warning: Tag '{target_tag}' has no corresponding English sample in atomic_data. Skipping the rest of this combination.")
            is_complete_combo = False
            break

    if not combined_content or not is_complete_combo:
        continue

    llm_input = {
        'optimal_combination': optimal_combination,
        'source_sample_ids': current_combo_sample_ids,
        'combined_content_for_llm': combined_content,
        **{k: v for k, v in combo_info.items() if k not in ['optimal_combination']}
    }

    llm_inputs.append(llm_input)

print(f"\nPrepared {len(llm_inputs)} LLM input samples (all English sources).")


if 'data' in locals():
    del data
    print("Deleted 'data' (Raw Parquet data).")

del atomic_data
print("Deleted 'atomic_data'.")

del ability_to_samples
print("Deleted 'ability_to_samples' (Index).")

del loaded_data
print("Deleted 'loaded_data' (Optimal Combinations).")

gc.collect()
print("Called garbage collector.")
print("--- Memory Cleanup Complete ---")

final_generated_data: List[Dict[str, Any]] = []
successful_generations = 0

with open(final_output_path, 'w', encoding='utf-8') as outfile:
    for i, llm_input in enumerate(tqdm(llm_inputs, desc="Openai Api Processing")):
        messages: List[Dict[str, str]] = []

        combined_content_blocks = llm_input['combined_content_for_llm']
        messages.append({"role": "system", "content": system_prompt})

        content_str = ""
        for item in combined_content_blocks:
            role = item.get('role', 'unknown')
            content = item.get('content', '')
            if role == 'user':
                content_str += f"user: {content}\n"
            elif role == 'assistant':
                content_str += f"assistant: {content}\n"
            elif role == 'skill_tag':
                content_str += f"\n--- {content} ---\n"
            else:
                content_str += f"{role.upper()}: {content}\n"

        messages.append({"role": "user", "content": content_str})

        generated_str = gen_response(messages)

        if generated_str:
            generated_content = string_to_builtin_type(generated_str)
            final_sample = {
                'id': f"generated_{successful_generations}_{os.getpid()}_{i}",
                'label': {'ability': llm_input['optimal_combination']},
                'content': generated_content,
                'source_info': {
                    'optimal_combination': llm_input['optimal_combination'],
                    'source_ids': llm_input['source_sample_ids'],
                    'max_conditional_se': llm_input.get('max_conditional_se'),
                }
            }

            outfile.write(json.dumps(final_sample, ensure_ascii=False) + '\n')
            final_generated_data.append(final_sample)
            successful_generations += 1
            if i % 100 == 0:
                final_generated_data = []
                print(
                    f"Processing progress: {i}/{len(llm_inputs)} samples processed. Successes: {successful_generations}.")
        if successful_generations >= args.gen_num:
            break

    print(f"\n======== 任务完成 (英文复合生成) ========")
    print(f"总共尝试生成 {len(llm_inputs)} 条数据。")
    print(f"成功生成并保存 {successful_generations} 条复杂数据到 {final_output_path}")