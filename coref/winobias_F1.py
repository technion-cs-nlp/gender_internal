import argparse
import os
import subprocess

import numpy as np
import wandb

dev_pro_results = []
dev_anti_results = []
dev_diff_results = []
test_pro_results = []
test_anti_results = []
test_diff_results = []

pro_results = []
anti_results = []
diff_results = []

def parse_args():
	parser = argparse.ArgumentParser(description='Test coreference model on winobias.')
	# parser.add_argument('--type', '-t', required=True, choices=[1, 2], type=int, help='the winobias type test with')
	parser.add_argument('--model_number', '-n', required=True, type=int, help='the model number to test')
	parser.add_argument('--model_name', '-m', required=True, choices=["balanced", "imbalanced", "anon", "CA"], type=str, help='the model to test')
	parser.add_argument('--retrain_seed', '-s', default=None, required=False, type=int, help='if model to be tested is a re-trained model,'
																		  'then specify the seed used for re-training')

	args = parser.parse_args()
	return args

def run(number, stereo, args):

	# # copy eval files to general file
	# subprocess.Popen(
	# 	f"cp winobias/test_type{type}_{stereo}_stereotype.128.jsonlines winobias/test.english.128.jsonlines".split())
	# subprocess.Popen(
	# 	f"cp winobias/dev_type{type}_{stereo}_stereotype.128.jsonlines winobias/dev.english.128.jsonlines".split())

	# copy eval files to general file
	subprocess.Popen(
		f"cp winobias/{stereo}_stereotype.128.jsonlines winobias/test.english.128.jsonlines".split())
	subprocess.Popen(
		f"cp winobias/{stereo}_stereotype.128.jsonlines winobias/dev.english.128.jsonlines".split())

	# delete cached tensors
	subprocess.Popen(f"rm winobias/cached.tensors.english.128.11.bin".split())

	# run evaluation

	if args.retrain_seed is not None:
		bashCommand = f"python evaluate.py test_roberta_base_{stereo}_retrain {args.model_name}{number}_seed{args.retrain_seed} 0"
		log_dir = f"winobias/test_roberta_base_{stereo}_retrain"
	else:
		# bashCommand = f"python evaluate.py test_roberta_base_t{type}_{stereo} {args.model_name}{number} 0"
		bashCommand = f"python evaluate.py test_roberta_base_{stereo} {args.model_name}{number} 0"
		log_dir = f"winobias/test_roberta_base_{stereo}"
	print("Running command:", bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	log_files = os.listdir(log_dir)
	log_files = list(map(lambda x: log_dir + "/" + x, log_files))
	log_file = sorted(log_files, key=os.path.getmtime)[-1]

	print(log_file)
	with open(log_file) as f:
		content = f.readlines()

	relevant = []
	for l in content:
		if l.startswith("Official avg F1: "):
			relevant.append(l)

	print(relevant)
	return float(relevant[0].strip().split()[-1])
	# return float(relevant[0].strip().split()[-1]), float(relevant[1].strip().split()[-1])

def log_results(pro, anti):
	# dev_pro_results.append(dev_pro)
	# dev_anti_results.append(dev_anti)
	# test_pro_results.append(test_pro)
	# test_anti_results.append(test_anti)
	pro_results.append(pro)
	anti_results.append(anti)
	diff_results.append(pro - anti)

def log_F1_gaps(args):

	if args.retrain_seed is not None:
		# dev_pro, test_pro = run(args.model_number, "pro", args.type, seed, args)
		pro = run(args.model_number, "pro", args)
		# dev_anti, test_anti = run(args.model_number, "anti", args.type, seed, args)
		anti = run(args.model_number, "anti", args)
		# log_results(dev_pro, dev_anti, test_pro, test_anti)
		log_results(pro, anti)
	else:
		# dev_pro_t1, test_pro_t1 = run(args.model_number, "pro", 1, args)
		pro = run(args.model_number, "pro", args)
		anti = run(args.model_number, "anti", args)
		# log_results(pro, anti)

	wandb.summary["F1_pro"] = pro
	wandb.summary["F1_anti"] = anti
	wandb.summary["F1_diff"] = pro - anti

	# wandb.summary["dev_F1_pro_mean"] = np.mean(dev_pro_results)
	# wandb.summary["dev_F1_pro_std"] = np.std(dev_pro_results)
	# wandb.summary["dev_F1_pro_all"] = dev_pro_results
	# wandb.summary["dev_F1_anti_mean"] = np.mean(dev_anti_results)
	# wandb.summary["dev_F1_anti_std"] = np.std(dev_anti_results)
	# wandb.summary["dev_F1_anti_all"] = dev_anti_results
	# wandb.summary["dev_F1_diff_mean"] = np.mean(dev_diff_results)
	# wandb.summary["dev_F1_diff_std"] = np.std(dev_diff_results)
	# wandb.summary["dev_F1_diff_all"] = dev_diff_results
	# wandb.summary["test_F1_pro_mean"] = np.mean(test_pro_results)
	# wandb.summary["test_F1_pro_std"] = np.std(test_pro_results)
	# wandb.summary["test_F1_pro_all"] = test_pro_results
	# wandb.summary["test_F1_anti_mean"] = np.mean(test_anti_results)
	# wandb.summary["test_F1_anti_std"] = np.std(test_anti_results)
	# wandb.summary["test_F1_anti_all"] = test_anti_results
	# wandb.summary["test_F1_diff_mean"] = np.mean(test_diff_results)
	# wandb.summary["test_F1_diff_std"] = np.std(test_diff_results)
	# wandb.summary["test_F1_diff_all"] = test_diff_results


def main():
	args = parse_args()
	#
	wandb.init(project="winobias", config={
		"architecture": "RoBERTa",
		"trained_on": "Ontonotes",
		"training balancing": args.model_name,
		# "type": args.type,
		# "n_models": 5,
		"retrain": args.retrain_seed,
		"metric": "F1",
		"model_number": args.model_number
	})

	log_F1_gaps(args)

if __name__ == "__main__":
	main()