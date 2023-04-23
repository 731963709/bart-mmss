import os
import utils
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

ori_inters = []
token_ids = {}
test_set = utils.BiDataset_t(os.path.join("../MINE_data", "test"))
word_ids = 0
for ori_src, ori_tgt in test_set:
    ori_inter = utils.get_intersection(ori_src, ori_tgt).split(" ")
    for word in ori_inter:
        if word not in token_ids:
            token_ids[word] = word_ids
            word_ids += 1
print("word size: ", len(token_ids))

for ori_src, ori_tgt in test_set:
    ori_inter = utils.get_intersection(ori_src, ori_tgt).split(" ")
    raw_arr = np.array([0 for _ in range(len(token_ids))])
    for word in ori_inter:
        if word in token_ids:
            raw_arr[token_ids[word]] = 1
    ori_inters.append(raw_arr)



for theta in [0]:
    filename = f"v20_33333_b16p8_theta{theta}"
    print(f"now {filename}")
    inter_strs = []
    counts = []
    with open(os.path.join("../experiments_MINE", filename, "test_candidate_inter.txt"), 'r') as fin:
        for line in fin:
            line = line.strip().split(' ')
            raw_arr = np.array([0 for _ in range(len(token_ids))])
            for word in line:
                if word in token_ids:
                    raw_arr[token_ids[word]]=1
            inter_strs.append(raw_arr)
            counts.append(np.sum(raw_arr))
    
    f1score = f1_score(ori_inters, inter_strs, average="macro", zero_division=0)
    precisionscore = precision_score(ori_inters, inter_strs, average="macro", zero_division=0)
    recallscore = recall_score(ori_inters, inter_strs, average="macro", zero_division=0)
    print(f"{theta} count {np.average(counts)} f1_score (%): {f1score*100} precision_score: {precisionscore*100} recall_score: {recallscore*100}")
    print("{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(np.average(counts), precisionscore*100, recallscore*100, f1score*100))