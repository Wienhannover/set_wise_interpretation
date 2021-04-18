# set_wise_interpretation

This responsitory is used to generate set-wise interpretations for predictions of Learning-to-rank(LTR) models. The proposed method ''pairwise INVASE with margin loss'' is based on INVASE. This dictionary contains the implementaions of proposed method and baselines.

# example command

python model_name --lamb 0.04 --margin 0.5 --patience 50 --start_stop 1000 --no_sample 6

# evaluation

The evaluation contains three points: the first point is to calculate the average number of selected features per instance, the second point is to calculate the Jaccard Similarity in one query and cross all queries, and the third point is to calculate the NDCG score of predictor network and baseline network.

python 1_selected_number_actor.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name

python 2_Jaccard_select_0.5.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name

python 3_mq2008_ndcg_rank_original_label.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name --ndcg_num 0 --ndcg_rank_num ndcg_cut_10






