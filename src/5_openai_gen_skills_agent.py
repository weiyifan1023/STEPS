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
import threading
import pandas as pd
from typing import List, Dict, Any, Set, Union
from collections import Counter
from openai import OpenAI
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=7, help="k value for file paths")
parser.add_argument("--gen_num", type=int, default=100, help="number of generated data")
parser.add_argument("--max_workers", type=int, default=5, help="number of threads")
args = parser.parse_args()


base_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/agent/k={args.k}"
final_output_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/agent/agent_skills_k={args.k}.jsonl"
tag_path = "/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/infinityinstruct_Gen"
atomic_data_path = "/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/atomic_data_infinity_en.jsonl"

os.makedirs(os.path.dirname(final_output_path), exist_ok=True)


def get_client():
    """初始化 OpenAI 客户端"""
    return OpenAI(
        api_key="",
        base_url="")


def gen_response(messages, max_retries: int = 3, initial_sleep: int = 1):
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=8192,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_sleep * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                return None


@lru_cache(maxsize=2048)
def is_agentic_combination(ability_tuple: tuple) -> bool:
    client = get_client()
    abilities = list(ability_tuple)
    filter_messages = [
        {"role": "system",
         "content": "You are a data quality auditor. Determine if the given skills can be combined into a complex, goal-oriented 'AI Agent' workflow. Reject scenarios that are just simple QA, pure creative writing, or basic translation."},
        {"role": "user",
         "content": f"Skills: {abilities}\nDoes this set have high potential for an autonomous agent workflow? Respond ONLY with 'YES' or 'NO'."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=filter_messages,
            max_tokens=10,
            temperature=0.0
        )
        decision = response.choices[0].message.content.strip().upper()
        return "YES" in decision
    except Exception as e:
        print(f"Filtering Error: {e}")
        return True


def string_to_builtin_type(data_string: str) -> Union[Any, str]:
    stripped_string = data_string.strip()
    if not stripped_string:
        return data_string
    try:
        if stripped_string.startswith("```python"):
            stripped_string = stripped_string[9:-3].strip()
        elif stripped_string.startswith("```"):
            stripped_string = stripped_string[3:-3].strip()
        return ast.literal_eval(stripped_string)
    except (ValueError, SyntaxError):
        return data_string

system_prompt = r"""You are the "Autonomous Agent Logic Architect," an expert in orchestrating high-fidelity cognitive trajectories for Multi-Skill AI Agents. Your mission is to transform disparate atomic skills into robust, data-driven agentic workflows.

Core Task: "Agentic Skill Synthesis"
1. Tactical Mapping: Analyze the provided Atomic Skill Tags and their source fragments. Define a "High-Value Objective" that no single skill could solve alone.
2. Agentic Persona & Environment: Construct a sophisticated professional environment where the Assistant acts as an "Act-then-Reflect" Agent.
3. Logical Integration (The Chain of Thought): The assistant must analyze constraints, plan a multi-step strategy, and execute using the technical lexicon of all integrated skills.
4. Data-Driven Instructions: The 'user' content must be a DIRECT, unified prompt. It MUST contain explicit, concrete data (e.g., a multi-row CSV snippet, a structured JSON object,a markdown table, a specific code block, or a technical log). If the source fragments are abstract, you MUST FABRICATE realistic, detailed data to populate the instruction so the Assistant has specific values to process, calculate, or analyze. 
5. Hard-Technical Interweaving: Ensure the response fuses the skills (e.g., using 'API Design' to solve a 'Quantum Physics' telemetry problem) rather than addressing them sequentially. The 'assistant' must refer to the specific data provided in the 'user' prompt during execution.

Strict Output Format:
* Output ONLY a single Python list containing exactly one pair of turn: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}].
* The 'user' content must be the final synthesized instruction only, containing both the scenario context and the necessary data blocks (tables/code/JSON). 
* NO meta-commentary like "Atomic Skill Tags:" or "Constraint:".
* All text content within the dictionaries must be in English.
* NO markdown code blocks, NO preamble, NO postscript. 
* The content within the dictionaries must be raw text, properly escaped for Python string compatibility.
"""

file_paths = glob.glob(f"{base_path}/optimal_combinations_*.pkl")
file_paths.sort(key=lambda x: int(re.search(r'optimal_combinations_(\d+)\.pkl', x).group(1)))

loaded_data = []
for file_path in file_paths:
    try:
        with open(file_path, 'rb') as f:
            file_data = pickle.load(f)
        loaded_data.extend(file_data)
        print(f"Loaded {len(file_data)} combinations from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

random.shuffle(loaded_data)

atomic_data = []
if os.path.exists(atomic_data_path):
    print(f"Loading existing atomic_data from {atomic_data_path}...")
    with open(atomic_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                atomic_data.append(json.loads(line))
else:
    print("Atomic_data not found. Processing raw Parquet files...")
    parquet_files = glob.glob(os.path.join(tag_path, "*.parquet"))
    all_dfs = [pd.read_parquet(f, columns=['id', 'conversations', 'label', 'langdetect']) for f in parquet_files]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df[combined_df['langdetect'] == 'en'].copy()

    tag_counts = Counter()
    with open(atomic_data_path, 'w', encoding='utf-8') as outfile:
        for index, row in combined_df.iterrows():
            abilities = row['label'].get('ability_zh', [])
            if hasattr(abilities, 'tolist'): abilities = abilities.tolist()
            if not abilities: continue
            ability_counts = {tag: tag_counts[tag] for tag in abilities}
            min_c = min(ability_counts.values()) + random.randint(15, 25)
            choices = [t for t, c in ability_counts.items() if c <= min_c]
            selected_ability = random.choice(choices)
            tag_counts[selected_ability] += 1
            conv_list = [{'role': 'user' if c.get('from') == 'human' else 'assistant', 'content': c.get('value', '')}
                         for c in row['conversations']]
            new_sample = {'id': row['id'], 'label': {'ability': [selected_ability]}, 'content': conv_list,
                          'single_ability_tag': selected_ability}
            atomic_data.append(new_sample)
            outfile.write(json.dumps(new_sample, ensure_ascii=False) + '\n')


ability_to_samples = {}
for sample in atomic_data:
    tag = sample.get('single_ability_tag')
    if tag:
        if tag not in ability_to_samples: ability_to_samples[tag] = []
        ability_to_samples[tag].append(sample)

llm_inputs = []
print(f"Starting Robust Matching for k={args.k}...")

all_available_tags = list(ability_to_samples.keys())

for combo_info in loaded_data:
    original_combination = [combo_info['start_ability']] + combo_info['optimal_combination']

    combined_content = []
    current_ids = []
    final_tags = []

    for tag in original_combination:
        target_tag = tag.strip()

        if target_tag not in ability_to_samples or not ability_to_samples[target_tag]:
            target_tag = random.choice(all_available_tags)

        samples_pool = ability_to_samples[target_tag]
        available = [s for s in samples_pool if s['id'] not in current_ids]

        selected = random.choice(available if available else samples_pool)

        current_ids.append(selected['id'])
        final_tags.append(target_tag)
        combined_content.append({"role": "skill_tag", "content": f"The Skill involved is {target_tag}"})
        combined_content.extend(selected.get('content', []))

    if len(combined_content) > 0:
        llm_inputs.append({
            'optimal_combination': final_tags,
            'source_sample_ids': current_ids,
            'combined_content_for_llm': combined_content,
            'max_conditional_se': combo_info.get('max_conditional_se')
        })

    if len(llm_inputs) >= args.gen_num * 5:
        break

print(f"Robust Match Complete: Prepared {len(llm_inputs)} candidates.")


llm_inputs = []
for combo_info in loaded_data:
    optimal_combination = [combo_info['start_ability']] + combo_info['optimal_combination']
    combined_content = []
    current_ids = []
    is_complete = True
    for tag in optimal_combination:
        if tag in ability_to_samples and ability_to_samples[tag]:
            available = [s for s in ability_to_samples[tag] if s['id'] not in current_ids]
            selected = random.choice(available if available else ability_to_samples[tag])
            current_ids.append(selected['id'])
            combined_content.append({"role": "skill_tag", "content": f"The Skill involved is {tag}"})
            combined_content.extend(selected.get('content', []))
        else:
            is_complete = False
            break
    if is_complete:
        llm_inputs.append({
            'optimal_combination': optimal_combination,
            'source_sample_ids': current_ids,
            'combined_content_for_llm': combined_content,
            'max_conditional_se': combo_info.get('max_conditional_se')
        })

del atomic_data, ability_to_samples, loaded_data
gc.collect()



successful_generations = 0
counter_lock = threading.Lock()


def worker_process(item, idx):
    """单个任务的处理逻辑"""
    global successful_generations

    # 提前退出检查
    if successful_generations >= args.gen_num:
        return None

    # A. 能力过滤
    ability_tuple = tuple(sorted(item['optimal_combination']))
    if not is_agentic_combination(ability_tuple):
        return None

    # B. 组装生成 Prompt
    content_str = ""
    for block in item['combined_content_for_llm']:
        role = block.get('role', 'unknown')
        content = block.get('content', '')
        if role == 'skill_tag':
            content_str += f"\n--- {content} ---\n"
        else:
            content_str += f"{role}: {content}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_str}
    ]

    # C. 调用 LLM
    generated_str = gen_response(messages)
    if not generated_str:
        return None

    generated_content = string_to_builtin_type(generated_str)

    # D. 线程安全计数与 ID 分配
    with counter_lock:
        if successful_generations >= args.gen_num:
            return None
        current_id_num = successful_generations
        successful_generations += 1

    return {
        'id': f"agent_gen_{current_id_num}_{os.getpid()}_{idx}",
        'label': {'ability': item['optimal_combination']},
        'content': generated_content,
        'source_info': {
            'source_ids': item['source_sample_ids'],
            'max_conditional_se': item.get('max_conditional_se'),
        }
    }


print(f"\n[Starting Multi-threaded Generation] Target: {args.gen_num} samples.")

with open(final_output_path, 'w', encoding='utf-8') as outfile:
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(worker_process, item, i) for i, item in enumerate(llm_inputs)]

        # 实时收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()

                    if successful_generations % 10 == 0 or successful_generations == 1:
                        print(f"Status: {successful_generations}/{args.gen_num} generated.")

                # 检查是否已达标，提前结束监听
                if successful_generations >= args.gen_num:
                    break
            except Exception as e:
                print(f"Worker Error: {e}")

print(f"\n======== Task Completed ========")
print(f"Final output saved to: {final_output_path}")