ROOT_DIR=/share/project/weiyifan/KG_RAG/llm_rlhf

export CUDA_VISIBLE_DEVICES="3,0,1,2,4,5,6,7"
export WANDB_BASE_URL="https://api.bandw.top"
# DeepSpeed Team

OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
#    OUTPUT=/share/project/weiyifan/KG_RAG/results/ckpt_cg/ours/Qwen2.5-7B
#    OUTPUT=/share/project/weiyifan/KG_RAG/results/ckpt_cg/skillmix/Qwen2.5-7B_alpaca52k
    OUTPUT=/share/project/weiyifan/KG_RAG/results/ckpt_cg/ours/Qwen2.5-7B_unconstrained

fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# /share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/Instruct-SkillMix-SDA/ism_sda_k2_4K_SFTforamt.jsonl
# /share/project/weiyifan/KG_RAG/results/coding_Tree/seed_infinityinstruct_Gen/clean_data_k=2.jsonl
#DataPath=/share/project/weiyifan/KG_RAG/kg_rag/codingTree_inference/llm_judge/if_data/alpaca52k/train_sft.jsonl
DataPath=/share/project/weiyifan/KG_RAG/results/coding_Tree/seed_full_graph_130w_dedup/clean_data_k=2.jsonl

deepspeed  main2.py \
    --deepspeed \
    --project_name Combinatorial_generalization \
    --experiment_name sft_Qwen2.5-7B \
    --data_path $DataPath \
    --model_name_or_path /share/project/models/Qwen2.5-7B \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --save_interval 5000 \
    --eval_interval 100 \
    --max_seq_len 3076 \
    --learning_rate 9.65e-6 \
    --loss_scale 0.0 \
    --loss_scale_window 100 \
    --weight_decay -1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps -1 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --output_dir $OUTPUT \
    --part_data_size -1 \
    | tee $OUTPUT/training.log \
