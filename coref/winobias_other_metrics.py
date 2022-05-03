import argparse
import json

import numpy as np
import torch
import wandb
from allennlp.fairness import Independence, Separation, Sufficiency
from sklearn import preprocessing

all_professions = {
    'CEO': 39,
    'accountant': 61,
    'analyst': 41,
    'assistant': 85,
    'attendant': 76,
    'auditor': 61,
    'baker': 65,
    'carpenter': 2,
    'cashier': 73,
    'chief': 27,
    'cleaner': 89,
    'clerk': 72,
    'constructionworker': 4,
    'cook': 38,
    'counselor': 73,
    'designer': 54,
    'developer': 20,
    'driver': 6,
    'editor': 52,
    'farmer': 22,
    'guard': 22,
    'hairdresser': 92,
    'housekeeper': 89,
    'janitor': 34,
    'laborer': 4,
    'lawyer': 35,
    'librarian': 84,
    'manager': 43,
    'mechanic': 4,
    'mover': 18,
    'nurse': 90,
    'physician': 38,
    'receptionist': 90,
    'salesperson': 48,
    'secretary': 95,
    'sheriff': 14,
    'supervisor': 44,
    'tailor': 80,
    'teacher': 78,
    'writer': 63
}
female_words = ["she", "her", "herself"]
male_words = ["he", "his", "him", "himself"]

def parse_args():
	parser = argparse.ArgumentParser(description='Test coreference model on winobias.')
	# parser.add_argument('--type', '-t', required=True, choices=[1, 2], type=int, help='the winobias type test with')
	parser.add_argument('--model_name', '-m', required=True, choices=["balanced", "imbalanced", "anon", "CA"], type=str, help='the model to test')
	parser.add_argument('--model_number', '-n', required=True, type=int, help='the model number to test')
	# parser.add_argument('--retrain_seeds', '-s', default=None, required=False, type=list, help='if model to be tested is a re-trained model,'
	# 																	  'then specify the seeds used for re-training')
	parser.add_argument('--retrain_seed', '-s', default=None, required=False, type=int, help='if model to be tested is a re-trained model,'
																		  'then specify the seed used for re-training')

	args = parser.parse_args()
	return args

def get_raw_percentage_and_gaps(all_professions, gaps):
	percentages = []
	values = []
	for name in all_professions:
		percentages.append(all_professions[name])
		values.append(gaps[name])

	return percentages, values

def log_classification_gaps(args):

	tpr_gaps_pearson = []
	tpr_gaps_abs_sum = []
	fpr_gaps_pearson = []
	fpr_gaps_abs_sum = []
	precision_gaps_pearson = []
	precision_gaps_abs_sum = []
	independence_sum = []
	separation_abs_sum = []
	# sufficiency_abs_sum = []
	sufficiency = []

	if args.retrain_seed:
		preds_pro = torch.load(f"../coref-hoi/preds/pred_{args.model_name}{args.model_number}_seed{args.retrain_seed}_test_roberta_base_pro_retrain.pt")
		preds_anti = torch.load(f"../coref-hoi/preds/pred_{args.model_name}{args.model_number}_seed{args.retrain_seed}_test_roberta_base_anti_retrain.pt")
		preds = list(preds_pro.values()) + list(preds_anti.values())
	else:
		preds_pro = torch.load(f"../coref-hoi/preds/pred_{args.model_name}{args.model_number}_test_roberta_base_pro.pt")
		preds_anti = torch.load(f"../coref-hoi/preds/pred_{args.model_name}{args.model_number}_test_roberta_base_anti.pt")
		preds = list(preds_pro.values()) + list(preds_anti.values())

	# data_test = load_gold_information(args.type, split)
	data_test = load_gold_information()

	genders, professions, predicted_professions = get_classification_array(data_test, preds)
	perc_values = []
	tpr_gaps_values = []
	fpr_gaps_values = []
	precision_gaps_values = []

	for prof in all_professions:
		metrics_male = metrics_fn(predicted_professions, professions, genders, prof, 'M')
		metrics_female = metrics_fn(predicted_professions, professions, genders, prof, 'F')

		tpr_gap = metrics_female['tpr'] - metrics_male['tpr']
		fpr_gap = metrics_female['fpr'] - metrics_male['fpr']
		precision_gap = metrics_female['precision'] - metrics_male['precision']

		perc_values.append(all_professions[prof])
		tpr_gaps_values.append(tpr_gap)
		fpr_gaps_values.append(fpr_gap)
		precision_gaps_values.append(precision_gap)

	# tpr_gaps_pearson.append(np.corrcoef(perc_values, tpr_gaps_values)[0][1])
	# fpr_gaps_pearson.append(np.corrcoef(perc_values, fpr_gaps_values)[0][1])
	# precision_gaps_pearson.append(np.corrcoef(perc_values, precision_gaps_values)[0][1])
	# tpr_gaps_abs_sum.append(np.sum(np.abs(tpr_gaps_values)))
	# fpr_gaps_abs_sum.append(np.sum(np.abs(fpr_gaps_values)))
	# precision_gaps_abs_sum.append(np.sum(np.abs(precision_gaps_values)))

	# +1 for label None
	metrics_allen = allennlp_metrics(len(all_professions) + 1, 2, predicted_professions, professions, genders)
	# independence_sum.append(metrics_allen["independence_sum"])
	# separation_abs_sum.append(metrics_allen["separation_gaps_abs_sum"])
	# sufficiency_abs_sum.append(metrics_allen["sufficiency_gaps_abs_sum"])
	# sufficiency.append(metrics_allen["sufficiency"])


	wandb.summary[f"tpr_gap_pearson"] = np.corrcoef(perc_values, tpr_gaps_values)[0][1]
	wandb.summary[f"fpr_gap_pearson"] = np.corrcoef(perc_values, fpr_gaps_values)[0][1]
	wandb.summary[f"precision_gap_pearson"] = np.corrcoef(perc_values, precision_gaps_values)[0][1]
	wandb.summary[f"tpr_gap_abs_sum"] = np.sum(np.abs(tpr_gaps_values))
	wandb.summary[f"fpr_gap_abs_sum"] = np.sum(np.abs(fpr_gaps_values))
	wandb.summary[f"precision_gap_abs_sum"] = np.sum(np.abs(precision_gaps_values))
	wandb.summary[f"independence_sum"] = metrics_allen["independence_sum"]
	wandb.summary[f"separation_gaps_abs_sum"] = metrics_allen["separation_gaps_abs_sum"]
	wandb.summary[f"sufficiency_gaps_abs_sum"] = metrics_allen["sufficiency_gaps_abs_sum"]
	wandb.summary[f"sufficiency"] = metrics_allen["sufficiency"]
	wandb.summary[f"separation"] = metrics_allen["separation"]
	wandb.summary[f"independence"] = metrics_allen["independence"]

	# wandb.summary[f"{split}_tpr_gap_pearson_mean"] = np.mean(tpr_gaps_pearson)
	# wandb.summary[f"{split}_tpr_gap_pearson_std"] = np.std(tpr_gaps_pearson)
	# wandb.summary[f"{split}_tpr_gap_pearson_all"] = tpr_gaps_pearson
	# wandb.summary[f"{split}_tpr_gap_abs_sum_mean"] = np.mean(tpr_gaps_abs_sum)
	# wandb.summary[f"{split}_tpr_gap_abs_sum_std"] = np.std(tpr_gaps_abs_sum)
	# wandb.summary[f"{split}_tpr_gap_abs_sum_all"] = tpr_gaps_abs_sum
	#
	# wandb.summary[f"{split}_fpr_gap_pearson_mean"] = np.mean(fpr_gaps_pearson)
	# wandb.summary[f"{split}_fpr_gap_pearson_std"] = np.std(fpr_gaps_pearson)
	# wandb.summary[f"{split}_fpr_gap_pearson_all"] = fpr_gaps_pearson
	# wandb.summary[f"{split}_fpr_gap_abs_sum_mean"] = np.mean(fpr_gaps_abs_sum)
	# wandb.summary[f"{split}_fpr_gap_abs_sum_std"] = np.std(fpr_gaps_abs_sum)
	# wandb.summary[f"{split}_fpr_gap_abs_sum_all"] = fpr_gaps_abs_sum
	#
	# wandb.summary[f"{split}_precision_gap_pearson_mean"] = np.mean(precision_gaps_pearson)
	# wandb.summary[f"{split}_precision_gap_pearson_std"] = np.std(precision_gaps_pearson)
	# wandb.summary[f"{split}_precision_gap_pearson_all"] = precision_gaps_pearson
	# wandb.summary[f"{split}_precision_gap_abs_sum_mean"] = np.mean(precision_gaps_abs_sum)
	# wandb.summary[f"{split}_precision_gap_abs_sum_std"] = np.std(precision_gaps_abs_sum)
	# wandb.summary[f"{split}_precision_gap_abs_sum_all"] = precision_gaps_abs_sum
	#
	# wandb.summary[f"{split}_independence_sum_mean"] = np.mean(independence_sum)
	# wandb.summary[f"{split}_independence_sum_std"] = np.std(independence_sum)
	# wandb.summary[f"{split}_independence_sum_all"] = independence_sum
	#
	# wandb.summary[f"{split}_separation_abs_sum_mean"] = np.mean(separation_abs_sum)
	# wandb.summary[f"{split}_separation_abs_sum_std"] = np.std(separation_abs_sum)
	# wandb.summary[f"{split}_separation_abs_sum_all"] = separation_abs_sum

	# wandb.summary[f"{split}_sufficiency_abs_sum_mean"] = np.mean(sufficiency_abs_sum)
	# wandb.summary[f"{split}_sufficiency_abs_sum_std"] = np.std(sufficiency_abs_sum)
	# wandb.summary[f"{split}_sufficiency_abs_sum_all"] = sufficiency_abs_sum

	# wandb.summary[f"{split}_sufficiency"] = sufficiency

def dictionary_torch_to_number(d: dict):
	for k, v in d.items():
		if isinstance(v, dict):
			dictionary_torch_to_number(v)
		else:
			d[k] = v.item()

def allennlp_metrics(n_labels, n_protecred_attributes, y_pred, y, z):

	le_gender = preprocessing.LabelEncoder()
	z = le_gender.fit_transform(z)
	z = torch.from_numpy(z)

	# For separation we ignore "None" predictions
	not_none_idx = np.where(y_pred != 'None')
	y_pred_sep = y_pred[not_none_idx]
	z_pred_sep = z[not_none_idx]
	y_sep = y[not_none_idx]
	le_sep = preprocessing.LabelEncoder()
	y_pred_sep = le_sep.fit_transform(y_pred_sep)
	y_pred_sep = torch.from_numpy(y_pred_sep)
	y_sep = le_sep.transform(y_sep)
	y_sep = torch.from_numpy(y_sep)

	le = preprocessing.LabelEncoder()
	y_pred = le.fit_transform(y_pred)
	y_pred = torch.from_numpy(y_pred)
	y = le.transform(y)
	y = torch.from_numpy(y)

	independence = Independence(n_labels, n_protecred_attributes)
	independence(y_pred, z)
	independence_score = independence.get_metric()

	separation = Separation(n_labels - 1, n_protecred_attributes)
	separation(y_pred_sep, y_sep, z_pred_sep)
	separation_score = separation.get_metric()

	sufficiency = Sufficiency(n_labels, n_protecred_attributes, dist_metric="wasserstein")
	sufficiency(y_pred, y, z)
	sufficiency_score = sufficiency.get_metric()

	dictionary_torch_to_number(independence_score)
	dictionary_torch_to_number(separation_score)
	dictionary_torch_to_number(sufficiency_score)

	separation_gaps = [scores[0] - scores[1] for label, scores in sorted(separation_score.items())]  # positive value - more separation for women
	sufficiency_gaps = [scores[0] - scores[1] for label, scores in sorted(sufficiency_score.items())]

	# Hack of sufficiency gaps that are is Nan
	# if np.isnan(sufficiency_gaps).any():
	# 	sufficiency_gaps = np.array(sufficiency_gaps)
	# 	max_value = np.max(sufficiency_gaps[~np.isnan(sufficiency_gaps)])
	# 	sufficiency_gaps[np.isnan(sufficiency_gaps)] = max_value

	return {"independence": json.dumps(independence_score), "separation": json.dumps(separation_score),
			"sufficiency": json.dumps(sufficiency_score),
			"independence_sum": independence_score[0] + independence_score[1],
			"separation_gaps_abs_sum": np.sum(np.abs(separation_gaps)),
			"sufficiency_gaps_abs_sum": np.sum(np.abs(sufficiency_gaps))}

def metrics_fn(y_pred, golden_y, z, label: str, gender: str):
	assert (len(y_pred) == len(golden_y))

	tp_indices = np.logical_and(np.logical_and(z == gender, golden_y == label),
								y_pred == label)  # only correct predictions of this gender

	n_tp = np.sum(tp_indices).item()
	pos_indices = np.logical_and(z == gender, golden_y == label)
	n_pos = np.sum(pos_indices).item()
	tpr = n_tp / n_pos

	fp_indices = np.logical_and(np.logical_and(z == gender, golden_y != label), y_pred == label)
	neg_indices = np.logical_and(z == gender, golden_y != label)
	n_fp = np.sum(fp_indices)
	n_neg = np.sum(neg_indices)
	fpr = n_fp / n_neg

	n_total_examples = len(y_pred)
	precision = n_tp / n_total_examples

	return {"tpr": tpr, "fpr": fpr, "precision": precision}

def load_gold_information():

	def load_jsonlines(f):
		data = []
		with open(f) as f:
			for line in f:
				data.append(json.loads(line))
		return data

	# data = load_jsonlines(f"../coref-hoi/winobias/{split}_type{t}_pro_stereotype.128.jsonlines") + load_jsonlines(
	# 	f"../coref-hoi/winobias/{split}_type{t}_anti_stereotype.128.jsonlines")
	data = load_jsonlines(f"../coref-hoi/winobias/pro_stereotype.128.jsonlines") + load_jsonlines(
		f"../coref-hoi/winobias/anti_stereotype.128.jsonlines")
	return data

def get_classification_array(data, preds):

	def get_word(sentence, span):
		subwords = sentence[span[0]: span[1] + 1]
		if "the" in subwords or "The" in subwords or "A" in subwords or "a" in subwords:
			subwords = subwords[1:]
		word = "".join(subwords)
		return word


	def intersection(list1, list2):
		inter = []
		for w2 in list2:
			for w1 in list1:
				if w2 in w1:
					inter.append(w2)
		return inter


	genders = []
	professions = []
	predicted_professions = []
	for ex, pred in zip(data, preds):
		for span in ex['clusters'][0]:
			word = get_word(ex['sentences'][0], span)
			if word in female_words:
				gender = 'F'
			elif word in male_words:
				gender = 'M'
			else:
				professions.append(word)
		genders.append(gender)

		pred_prof = 'None'
		for cluster in pred:
			words = []
			for span in cluster:
				word = get_word(ex['sentences'][0], span)
				words.append(word)

			inter = intersection(words, all_professions)
			if (len(intersection(words, female_words + male_words)) > 0) and (len(inter) == 1):
				pred_prof = inter[0]
		predicted_professions.append(pred_prof)

	genders = np.array(genders)
	professions = np.array(professions)
	predicted_professions = np.array(predicted_professions)

	return genders, professions, predicted_professions

def main():
	args = parse_args()
	#
	wandb.init(project="winobias", config={
		"architecture": "RoBERTa",
		"trained_on": "Ontonotes",
		"training balancing": args.model_name,
		# "type": args.type,
		"retrain": args.retrain_seed,
		"metric": "other",
		"model_number": args.model_number
	})

	log_classification_gaps(args)
	# log_classification_gaps(args, "test")
	# log_classification_gaps(args, "dev")

if __name__ == "__main__":
	main()