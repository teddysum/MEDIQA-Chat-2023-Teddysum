CUDA_VISIBLE_DEVICES=0,1,2,3 python dialogLED_base_sum.py \
  --base_model MingZhong/DialogLED-large-5120 \
  --do_inference \
  --model_path ../models/dialogLED_base_sum.pt \
  --batch_size 4 \
  --max_src_len 5120 \
  --test_data $1 \
  --output_dir ../result/ \
  --output_file_name taskB_Teddysum_run1.csv

#  ../data/TaskB/taskB_testset4participants_inputConversations.csv