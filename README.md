# CUFL-RG
python main_epinions.py --embed_size=4 --lr=0.1 --data=epinions --user_batch=256 --clip=0.1 --ldp_epsilon=3 --negative_sample=10 --valid_step=20 --weight_decay=0.001 --device=cuda:0 --rounds=1500 --social_reg=0.03 --contrastive_temp=0.1 --contrastive_neg_num=16 --item_contrastive_reg=0.01 --item_contrastive_temp=0.175 --item_contrastive_neg_num=8 


python main_ciaodvd.py --embed_size=8 --lr=0.1 --data=ciaodvd --user_batch=256 --clip=0.4 --laplace_lambda=0.4 --negative_sample=150 --valid_step=30 --weight_decay=0.001 --device=cuda:0 --rounds=500 --social_reg=0.03 --contrastive_temp=0.2 --contrastive_neg_num=16 --item_contrastive_reg=0.01 --item_contrastive_temp=0.2 --item_contrastive_neg_num=8


python main_ciao.py --embed_size=32 --lr=0.01 --data=ciao --user_batch=1024 --clip=0.1 --laplace_lambda=0.3 --negative_sample=10 --valid_step=20 --weight_decay=0.001 --device=cuda:0 --rounds=1000 --social_reg=0.03 --contrastive_temp=0.2 --contrastive_neg_num=16 --item_contrastive_reg=0.01 --item_contrastive_temp=0.2 --item_contrastive_neg_num=8

