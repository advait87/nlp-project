deepspeed --num_gpus 1 --module openrlhf.cli.train_sft \
   --pretrain Qwen/Qwen2.5-0.5B-Instruct \
   --dataset "./data/pgmix_stratified/processed" \
   --input_key "input" \
   --output_key "output" \
   --input_template "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.
In addition, identify the policies violated if the interaction is unsafe and harmful according to the safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

Human user:
{}

AI assistant:
" \
   --max_len 1024 \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --save_path ./checkpoint/polyguard-smol-v4 \
   --logging_steps 10 \
   --save_steps 100 \
   --zero_stage 2 \
   --adam_offload \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 2e-5 \
   --gradient_checkpointing \
   --lora_rank 16 \
   --lora_alpha 32 \
   --lora_dropout 0.1 \
   --seed 42