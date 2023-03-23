CUDA_VISIBLE_DEVICES=0,1,2,3 python dialogLED_prompt_base_sum_by_section.py \
  --base_model MingZhong/DialogLED-large-5120 \
  --do_inference \
  --model_path ../models/dialogLED_prompt_base_sum_by_section.pt \
  --batch_size 4 \
  --max_src_len 5120 \
  --test_data $1 \
  --output_dir ../result/ \
  --output_file_name taskB_Teddysum_run3.csv

  
#  ../data/TaskB/taskB_testset4participants_inputConversations.csv