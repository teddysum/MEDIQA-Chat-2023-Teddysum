
VAR="133qI3DKEXIydoPQGfS_1e7q1gSMjUHXt"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$VAR" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$VAR" -o "../models/dialogLED_base_sum.pt"
rm cookie

VAR="1HI7grpncSwYssa5izQhF96Xe0so53ZVe"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$VAR" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$VAR" -o "../models/dialogLED_prompt_base_sum_by_section.pt"
rm cookie

VAR="1oL_BrUIy-OjWOCdJgwp7_tPgmQxQA835"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$VAR" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$VAR" -o "../models/dialogLED_prompt_base_sum.pt"
rm cookie

conda init
conda create -n mediqa-teddysum-env

source /opt/conda/etc/profile.d/conda.sh
conda activate mediqa-teddysum-env

python3 -m venv evalvenv
source evalvenv/bin/activate
pip install -r requirements.txt
conda install pandas
conda install pytorch torchvision torchaudio -c pytorch
conda install transformers

conda deactivate
