import os
import json
import random
import re
from typing import List, Dict, Any

CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]')

def load_jsonl(file_path):
    """从指定路径加载 JSONL 文件。"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        dat = [json.loads(line) for line in f.readlines()]
    return dat


def save_jsonl(sample_ls, save_path):
    """将样本列表保存为 JSONL 文件。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for ipt in sample_ls:
            json_str = json.dumps(ipt, ensure_ascii=False)
            f.write(json_str + '\n')
    print(f"Saved {len(sample_ls)} samples to {save_path}")


def preprocess_and_filter_data1(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    预处理并过滤样本列表，确保 'content' 字段是符合SFT要求的对话列表格式，
    并且每个消息的 'content' 值必须是字符串，且**不包含中文字符**。
    """
    CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]')
    clean_samples = []
    filtered_count_chinese = 0

    for i, sample in enumerate(samples):
        if 'content' not in sample:
            continue

        content = sample['content']

        if not isinstance(content, list) or not content:
            continue

        is_valid_dialogue = True
        has_chinese_content = False

        for message in content:
            if not isinstance(message, dict):
                is_valid_dialogue = False
                break

            if 'role' not in message or 'content' not in message:
                is_valid_dialogue = False
                break

            message_content = message['content']

            if not isinstance(message_content, str):
                if isinstance(message_content, (int, float)):
                    message_content = str(message_content)
                    message['content'] = message_content
                else:
                    is_valid_dialogue = False
                    break

            if CHINESE_CHAR_PATTERN.search(message_content):
                has_chinese_content = True
                break

        if has_chinese_content:
            filtered_count_chinese += 1
            continue

        if is_valid_dialogue:
            clean_samples.append(sample)

    num_filtered_total = len(samples) - len(clean_samples)

    if num_filtered_total > 0:
        print(f"--- Filtered {num_filtered_total} samples in total. ---")
        if filtered_count_chinese > 0:
            print(f"  - {filtered_count_chinese} samples filtered due to containing Chinese characters.")
        other_filtered = num_filtered_total - filtered_count_chinese
        if other_filtered > 0:
            print(f"  - {other_filtered} samples filtered due to invalid dialogue format or non-string content.")

    return clean_samples

def preprocess_and_filter_data(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    预处理并过滤样本列表，确保 'content' 字段是符合SFT要求的对话列表格式，
    并且每个消息的 'content' 值必须是字符串。
    """
    clean_samples = []

    for i, sample in enumerate(samples):
        if 'content' not in sample:
            continue

        content = sample['content']

        if not isinstance(content, list) or not content:
            continue

        is_valid_dialogue = True

        for message in content:
            if not isinstance(message, dict):
                is_valid_dialogue = False
                break

            if 'role' not in message or 'content' not in message:
                is_valid_dialogue = False
                break

            message_content = message['content']
            if not isinstance(message_content, str):
                if isinstance(message_content, (int, float)):
                    message['content'] = str(message_content)
                else:
                    is_valid_dialogue = False
                    break

        if is_valid_dialogue:
            clean_samples.append(sample)

    num_filtered = len(samples) - len(clean_samples)
    if num_filtered > 0:
        print(f"--- Filtered {num_filtered} samples due to invalid dialogue or non-string message content. ---")

    return clean_samples


def load_and_group_data(base_path: str, k_values: List[int]) -> tuple[Dict[int, List[Dict[str, Any]]], int]:
    """
    加载所有 k 值的数据，并将每个样本添加 'k' 字段。
    在分组前会调用预处理和过滤函数。
    """
    data_dict = {}
    total_unique_samples = 0
    print("--- 1. Loading and Filtering Data ---")

    for k in k_values:
        file_name = f'generated_complex_data_k={k}.jsonl'
        file_path = os.path.join(base_path, file_name)

        raw_samples = load_jsonl(file_path)

        filtered_samples = preprocess_and_filter_data(raw_samples)

        samples_with_k = [{'k': k, **s} for s in filtered_samples]

        data_dict[k] = samples_with_k
        total_unique_samples += len(samples_with_k)
        print(f"Loaded and Filtered k={k}: {len(samples_with_k)} final samples.")

    return data_dict, total_unique_samples


def create_curriculum_schedule_flexible_sampling_V4(data_dict, k_values, num_epochs=3):
    """
    V4 修复版本：确保每个 Epoch 总样本量 N=6000。
    采用灵活采样策略（无重复/重复结合），使用索引追踪解决样本量失控问题。
    """
    random.seed(2024)
    epoch_data_list = []

    TARGET_SAMPLES_PER_EPOCH = 6000
    data_size_by_k = {k: len(data_dict[k]) for k in k_values}

    shuffled_unique_data_by_k = {
        k: random.sample(data_dict[k], data_size_by_k[k])
        for k in k_values
    }

    current_idx_by_k = {k: 0 for k in k_values}

    # 定义每个 k 在每个 Epoch 中占的“权重”或“比例因子”
    # 策略：k小权重递减，k大权重递增，总权重和为 1.00
    # k: 2, 3, 4, 5, 6, 7
    WEIGHTS_BY_EPOCH = {
        1: [0.60, 0.20, 0.10, 0.05, 0.03, 0.02],
        2: [0.35, 0.25, 0.15, 0.10, 0.08, 0.07],
        3: [0.20, 0.20, 0.20, 0.15, 0.13, 0.12],
    }

    print("\n--- 2. Creating Curriculum Schedule (V4: Fixed Sampling) ---")

    for epoch in range(1, num_epochs + 1):
        current_epoch_samples = []

        weights = WEIGHTS_BY_EPOCH[epoch]
        samples_to_take = {}
        for i, k in enumerate(k_values):
            count = round(weights[i] * TARGET_SAMPLES_PER_EPOCH)
            samples_to_take[k] = count

        actual_total = sum(samples_to_take.values())
        diff = TARGET_SAMPLES_PER_EPOCH - actual_total
        if diff != 0:
            samples_to_take[k_values[0]] += diff

        print(f"\n-- Epoch {epoch} Allocation (Target: {TARGET_SAMPLES_PER_EPOCH}): --")

        for k in k_values:
            count = samples_to_take[k]
            unique_size = data_size_by_k[k]
            start_idx = current_idx_by_k[k]

            samples_to_add = []

            if count <= unique_size:
                end_idx = start_idx + count

                if end_idx > unique_size:
                    samples_to_add.extend(shuffled_unique_data_by_k[k][start_idx:])
                    needed_from_start = end_idx - unique_size
                    samples_to_add.extend(shuffled_unique_data_by_k[k][:needed_from_start])

                    current_idx_by_k[k] = needed_from_start
                else:
                    samples_to_add.extend(shuffled_unique_data_by_k[k][start_idx:end_idx])
                    current_idx_by_k[k] = end_idx

                print(f"  k={k}: Taking {len(samples_to_add)} samples (Unique).")

            else:
                samples_to_add = random.choices(shuffled_unique_data_by_k[k], k=count)

                current_idx_by_k[k] = (start_idx + unique_size) % unique_size

                print(f"  k={k}: Taking {count} samples (REQUIRED REPEAT).")

            current_epoch_samples.extend(samples_to_add)

        random.shuffle(current_epoch_samples)

        final_len = len(current_epoch_samples)
        print(f"  Total samples for Epoch {epoch}: {final_len}")
        epoch_data_list.append(current_epoch_samples)

    return epoch_data_list


def clean_single_data(k=2):
    file_name = f'generated_complex_data_k={k}.jsonl'
    base_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen"
    file_path = os.path.join(base_path, file_name)
    save_path = f"/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/clean_data_k={k}.jsonl"

    raw_samples = load_jsonl(file_path)

    filtered_samples = preprocess_and_filter_data1(raw_samples)

    print(f"原始样本数: {len(raw_samples)}")
    print(f"有效样本数: {len(filtered_samples)}")
    save_jsonl(filtered_samples, save_path)
    return filtered_samples



if __name__ == "__main__":
    # clean_single_data()

    DATA_ROOT_PATH = '/share/project/weiyifan/KG_RAG/results/coding_Tree'
    SAVE_DIR = './schedule_data_cl'
    K_VALUES = list(range(2, 8))  # k=2, 3, 4, 5, 6, 7
    NUM_EPOCHS = 3
    TARGET_SAMPLES_PER_EPOCH = 6000

    data_dict, total_unique_N = load_and_group_data(DATA_ROOT_PATH, K_VALUES)

    print(f"\nTotal unique samples: {total_unique_N}")

    epoch_data_list = create_curriculum_schedule_flexible_sampling_V4(
        data_dict, K_VALUES, NUM_EPOCHS
    )


    print("\n--- 3. Saving Epoch Data Files ---")
    all_sch_sample = []
    for i, epoch_samples in enumerate(epoch_data_list):
        epoch_num = i + 1
        save_path = os.path.join(SAVE_DIR, f'cl_epoch_{epoch_num}_{str(TARGET_SAMPLES_PER_EPOCH)}.jsonl')
        save_jsonl(epoch_samples, save_path)
        all_sch_sample.extend(epoch_samples)

    # final_save_path = os.path.join(SAVE_DIR,
    #                                f'cl_all_epochs_{str(int(TARGET_SAMPLES_PER_EPOCH * NUM_EPOCHS))}.jsonl')
    # save_jsonl(all_sch_sample, final_save_path)
    #
    # print("\n✅ Flexible Curriculum Learning data schedule creation complete.")