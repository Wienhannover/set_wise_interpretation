# set_wise_interpretation

This responsitory is used to generate set-wise interpretations for predictions of Learning-to-rank(LTR) models. The set-wise interpretation means that a query has a commen interpretation for the corresponding documents, but different queries have different interpretations.  The proposed method ''pairwise INVASE with margin loss'' is based on [INVASE](https://github.com/jsyoon0823/INVASE#codebase-for-invase-instance-wise-variable-selection). This dictionary contains the implementaions of proposed method and the baselines.

# example command

model_name refers to different models, lamb is a hyperparameter used to control the number of selected features per instance, margin is a hyperparameter used to force selector to generate same output in one query and different output of differenr queries, no_sample refers to the number of negative instances that are used in the margin loss.
```
python model_name --lamb 0.04 --margin 0.5 --patience 50 --start_stop 1000 --no_sample 6
```
# evaluation

The evaluation contains three points: the first point is to calculate the average number of selected features per instance, the second point is to calculate the Jaccard Similarity in one query and cross all queries, and the third point is to calculate the NDCG score of predictor network and baseline network.
```
python evaluate/1_selected_number_actor.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name

python evaluate/2_Jaccard_select_0.5.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name

python evaluate/3_mq2008_ndcg_rank_original_label.py --data_path MQ2008 --feat_num 46 --model_path saved_model_name --ndcg_num 0 --ndcg_rank_num ndcg_cut_10
```





