conda create --name poinncare -c conda-forge -c pytorch -c nvidia --file requirements.txt
conda activate poinncare
source set_env.sh 

# parameter setting for 30 and 30-50 test sets
python -u train_align.py --preprocess-adj2 PPR --ppr-alpha2 0.8 --diff-layers2 2 --loss3_weight 0.001 --output1_weight 0.8 --output2_weight 0.2 --save-dir logs/30/ 

# parameter setting for the price test set
python -u train_align.py --preprocess-adj2 PPR --ppr-alpha2 0.8 --diff-layers2 2 --loss3_weight 0.0001 --loss4_weight 0.0001 --loss5_weight 0.0001 --output1_weight 0.2 --output2_weight 0.8 --min-epochs 5000 --save-dir logs/price_local/ 

# parameter setting for the promiscuous test set
python -u train_align.py --preprocess-adj1 PPR --ppr-alpha1 0.8 --diff-layers1 2 --preprocess-adj2 PPR --ppr-alpha2 0.8 --diff-layers2 2 --loss3_weight 0.0001 --save-dir logs/promiscuous_local/ 
