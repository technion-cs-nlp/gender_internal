import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer

def BiasinBios_split_and_return_tokens_data(seed, path, other=None, verbose=False):
    data = torch.load(path)
    X, y, att_masks, z = data["X"], data["y"], data["masks"], data["z"]

    cat = pd.Categorical(y)
    y = cat.codes

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, att_masks shape: {att_masks.shape}, z shape: {z.shape}")

    other_train = None
    other_valid = None
    other_test = None
    if other is None:
        X_train_valid, X_test, y_train_valid, y_test, att_masks_train_valid, att_masks_test, z_train_valid, z_test,\
            original_y_train_valid, original_y_test = train_test_split(
            X, y, att_masks, z, data["y"], random_state=seed, stratify=y, test_size=0.25)

        X_train, X_valid, y_train, y_valid, att_masks_train, att_masks_valid, z_train, z_valid, original_y_train,\
            original_y_valid = train_test_split(
            X_train_valid, y_train_valid, att_masks_train_valid, z_train_valid, original_y_train_valid, random_state=seed, stratify=y_train_valid,
            test_size=0.133)

    else:
        X_train_valid, X_test, y_train_valid, y_test, att_masks_train_valid, att_masks_test, z_train_valid, z_test, other_train_valid, other_test,\
            original_y_train_valid, original_y_test = train_test_split(
            X, y, att_masks, z, other, data["y"], random_state=seed, stratify=y, test_size=0.25)

        X_train, X_valid, y_train, y_valid, att_masks_train, att_masks_valid, z_train, z_valid, other_train, other_valid,\
            original_y_train, original_y_valid = train_test_split(
            X_train_valid, y_train_valid, att_masks_train_valid, z_train_valid, other_train_valid,
            original_y_train_valid, random_state=seed, stratify=y_train_valid,
            test_size=0.133)

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, att_masks_train shape: {att_masks_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, att_masks_valid shape: {att_masks_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, att_masks_test shape: {att_masks_test.shape}, z_test shape: {z_test.shape}")

    return {
        "categories": cat,
        "train":
            {
                "X": X_train,
                "y": y_train,
                "z": z_train,
                "original_y": original_y_train,
                "masks": att_masks_train,
                "other": other_train
            },
        "test":
            {
                "X": X_test,
                "y": y_test,
                "z": z_test,
                "original_y": original_y_test,
                "masks": att_masks_test,
                "other": other_test
            },
        "valid":
            {
                "X": X_valid,
                "y": y_valid,
                "z": z_valid,
                "original_y": original_y_valid,
                "masks": att_masks_valid,
                "other": other_valid
            }
    }

def BiasinBios_split_and_save_tokens_data(seed, type, verbose=False):
    path = f"../data/biosbias/tokens_{type}_roberta-base_128.pt"
    data = torch.load(path)
    X, y, att_masks, z = data["X"], data["y"], data["masks"], data["z"]

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, att_masks shape: {att_masks.shape}, z shape: {z.shape}")

    X_train_valid, X_test, y_train_valid, y_test, att_masks_train_valid, att_masks_test, z_train_valid, z_test = train_test_split(
        X, y, att_masks, z, random_state=seed, stratify=y, test_size=0.25)

    X_train, X_valid, y_train, y_valid, att_masks_train, att_masks_valid, z_train, z_valid = train_test_split(
        X_train_valid, y_train_valid, att_masks_train_valid, z_train_valid, random_state=seed, stratify=y_train_valid,
        test_size=0.133)

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, att_masks_train shape: {att_masks_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, att_masks_valid shape: {att_masks_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, att_masks_test shape: {att_masks_test.shape}, z_test shape: {z_test.shape}")

    torch.save({"X": X_train, "masks": att_masks_train, "y": y_train, "z": z_train},
               f"../data/biosbias/tokens_{type}_original_roberta-base_128_train-seed_{seed}.pt")
    torch.save({"X": X_valid, "masks": att_masks_valid, "y": y_valid, "z": z_valid},
               f"../data/biosbias/tokens_{type}_original_roberta-base_128_valid-seed_{seed}.pt")
    torch.save({"X": X_test, "masks": att_masks_test, "y": y_test, "z": z_test},
               f"../data/biosbias/tokens_{type}_original_roberta-base_128_test-seed_{seed}.pt")

def BiasinBios_split_and_return_vectors_data(seed, path, verbose=False):
    data = torch.load(path)
    X, y, z = data["X"], data["y"], data["z"]

    #tmp
    # X = torch.load("../data/biosbias/scrubbed_words_vector/data.pt")

    cat = pd.Categorical(y)
    y = cat.codes

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, z shape: {z.shape}")

    X_train_valid, X_test, y_train_valid, y_test, z_train_valid, z_test, original_y_train_valid, original_y_test = train_test_split(
        X, y, z, data["y"], random_state=seed, stratify=y, test_size=0.25)

    X_train, X_valid, y_train, y_valid, z_train, z_valid, original_y_train, original_y_valid = train_test_split(
        X_train_valid, y_train_valid, z_train_valid, original_y_train_valid, random_state=seed, stratify=y_train_valid,
        test_size=0.133)

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

    return {
        "categories": cat,
        "train":
            {
                "X": X_train,
                "y": y_train,
                "z": z_train,
                "original_y": original_y_train
            },
        "test":
            {
                "X": X_test,
                "y": y_test,
                "z": z_test,
                "original_y": original_y_test
            },
        "valid":
            {
                "X": X_valid,
                "y": y_valid,
                "z": z_valid,
                "original_y": original_y_valid
            }
    }

def BiasinBios_split_and_save_vectors_data(seed, type, verbose=False, folder="../data/biosbias"):
    path = folder + f"/vectors_{type}_roberta-base_128.pt"
    data = torch.load(path)
    # X, y, z, orig_idx = data["X"], data["y"], data["z"], data["original index"]
    X, y, z = data["X"], data["y"], data["z"]

    if verbose:
        print(f"X shape: {X.shape}, y shape: {y.shape}, z shape: {z.shape}")

    X_train_valid, X_test, y_train_valid, y_test, z_train_valid, z_test = train_test_split(
        X, y, z, random_state=seed, stratify=y, test_size=0.25)

    X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(
        X_train_valid, y_train_valid, z_train_valid, random_state=seed, stratify=y_train_valid,
        test_size=0.133)

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

    torch.save({"X": X_train, "y": y_train, "z": z_train},
               folder + f"/vectors_{type}_original_roberta-base_128_train-seed_{seed}.pt")
    torch.save({"X": X_valid, "y": y_valid, "z": z_valid},
               folder + f"/vectors_{type}_original_roberta-base_128_valid-seed_{seed}.pt")
    torch.save({"X": X_test, "y": y_test, "z": z_test},
               folder + f"/vectors_{type}_original_roberta-base_128_test-seed_{seed}.pt")

def BiasinBios_balance_tokens_dataset(type, seed, oversampling, verbose=False, folder="../data/biosbias"):
    data_train = torch.load(folder + f"/tokens_{type}_original_roberta-base_128_train-seed_{seed}.pt")
    data_valid = torch.load(folder + f"/tokens_{type}_original_roberta-base_128_valid-seed_{seed}.pt")
    data_test = torch.load(folder + f"/tokens_{type}_original_roberta-base_128_test-seed_{seed}.pt")

    X_train, y_train, z_train, masks_train = data_train["X"], data_train["y"], data_train["z"], data_train["masks"]
    X_valid, y_valid, z_valid, masks_valid = data_valid["X"], data_valid["y"], data_valid["z"], data_valid["masks"]
    X_test, y_test, z_test, masks_test = data_test["X"], data_test["y"], data_test["z"], data_test["masks"]

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, z_train shape: {z_train.shape}, masks_train shape: {masks_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, z_valid shape: {z_valid.shape}, masks_valid shape: {masks_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, z_test shape: {z_test.shape}, masks_test shape: {masks_test.shape}")

    X_train_new, y_train_new, z_train_new, masks_train_new = balance_dataset(X_train, y_train, z_train,
                                                                             masks=masks_train, oversampling=oversampling)
    X_valid_new, y_valid_new, z_valid_new, masks_valid_new = balance_dataset(X_valid, y_valid, z_valid,
                                                                             masks=masks_valid, oversampling=oversampling)
    X_test_new, y_test_new, z_test_new, masks_test_new = balance_dataset(X_test, y_test, z_test, masks=masks_test,
                                                                oversampling=oversampling)

    sampling = "oversampled" if oversampling else "subsampled"
    torch.save({"X": X_train_new, "masks": masks_train_new, "y": y_train_new, "z": z_train_new},
               folder + f"/tokens_{type}_{sampling}_roberta-base_128_train-seed_{seed}.pt")
    torch.save({"X": X_valid_new, "masks": masks_valid_new, "y": y_valid_new, "z": z_valid_new},
               folder + f"/tokens_{type}_{sampling}_roberta-base_128_valid-seed_{seed}.pt")
    torch.save({"X": X_test_new, "masks": masks_test_new, "y": y_test_new, "z": z_test_new},
               folder + f"/tokens_{type}_{sampling}_roberta-base_128_test-seed_{seed}.pt")

def BiasinBios_balance_vectors_dataset(type, seed, oversampling, verbose=False, folder="../data/biosbias"):
    data_train = torch.load(folder + f"/vectors_{type}_original_roberta-base_128_train-seed_{seed}.pt")
    data_valid = torch.load(folder + f"/vectors_{type}_original_roberta-base_128_valid-seed_{seed}.pt")
    data_test = torch.load(folder + f"/vectors_{type}_original_roberta-base_128_test-seed_{seed}.pt")

    X_train, y_train, z_train = data_train["X"], data_train["y"], data_train["z"]
    X_valid, y_valid, z_valid = data_valid["X"], data_valid["y"], data_valid["z"]
    X_test, y_test, z_test = data_test["X"], data_test["y"], data_test["z"]

    if verbose:
        print(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
        print(
            f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}, z_valid shape: {z_valid.shape}")
        print(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

    X_train_new, y_train_new, z_train_new = balance_dataset(X_train, y_train, z_train,
                                                                             oversampling=oversampling)
    X_valid_new, y_valid_new, z_valid_new = balance_dataset(X_valid, y_valid, z_valid,
                                                                             oversampling=oversampling)
    X_test_new, y_test_new, z_test_new = balance_dataset(X_test, y_test, z_test,
                                                                         oversampling=oversampling)

    sampling = "oversampled" if oversampling else "subsampled"
    torch.save({"X": X_train_new, "y": y_train_new, "z": z_train_new},
               folder + f"/vectors_{type}_{sampling}_roberta-base_128_train-seed_{seed}.pt")
    torch.save({"X": X_valid_new, "y": y_valid_new, "z": z_valid_new},
               folder + f"/vectors_{type}_{sampling}_roberta-base_128_valid-seed_{seed}.pt")
    torch.save({"X": X_test_new, "y": y_test_new, "z": z_test_new},
               folder + f"/vectors_{type}_{sampling}_roberta-base_128_test-seed_{seed}.pt")

def BiasinBios_balance_LM_dataset(type, seed, split_type, oversampling, verbose=False):
    data = torch.load(f"../data/biosbias/LM_tokens_{type}_original_roberta-base_128_{split_type}-seed_{seed}.pt")
    data2 = torch.load(f"../data/biosbias/LM_tokens_{type}_original_roberta-base_128_{split_type}_mask_length_2-seed_{seed}.pt")
    data3 = torch.load(f"../data/biosbias/LM_tokens_{type}_original_roberta-base_128_{split_type}_mask_length_3-seed_{seed}.pt")
    data4 = torch.load(f"../data/biosbias/LM_tokens_{type}_original_roberta-base_128_{split_type}_mask_length_4-seed_{seed}.pt")

    X, y, z, gold_sents, masks, true_length = \
        data["X"], data["labels"], data["z"], data["y"], data["masks"], data["true_length"]
    X2, y2, z2, gold_sents2, masks2, true_length2 = \
        data2["X"], data2["labels"], data2["z"], data2["y"], data2["masks"], data2["true_length"]
    X3, y3, z3, gold_sents3, masks3, true_length3 = \
        data3["X"], data3["labels"], data3["z"], data3["y"], data3["masks"], data3["true_length"]
    X4, y4, z4, gold_sents4, masks4, true_length4 = \
        data4["X"], data4["labels"], data4["z"], data4["y"], data4["masks"], data4["true_length"]

    if verbose:
        print(
            f"X shape: {X.shape}, y shape: {y.shape}, z shape: {z.shape}")

    X_new, y_new, z_new, gold_sents_new, masks_new, true_length_new = balance_dataset(X, y, z, other=(gold_sents, masks, true_length), oversampling=oversampling)
    X2_new, y2_new, z2_new, gold_sents2_new, masks2_new, true_length2_new = balance_dataset(X2, y2, z2, other=(gold_sents2, masks2, true_length2), oversampling=oversampling)
    X3_new, y3_new, z3_new, gold_sents3_new, masks3_new, true_length3_new = balance_dataset(X3, y3, z3, other=(gold_sents3, masks3, true_length3), oversampling=oversampling)
    X4_new, y4_new, z4_new, gold_sents4_new, masks4_new, true_length4_new = balance_dataset(X4, y4, z4, other=(gold_sents4, masks4, true_length4), oversampling=oversampling)

# "X": sents, "masks": masks, "y": gold_sents, "z": genders, "true_length": lengths, "labels": labels
    sampling = "oversampled" if oversampling else "subsampled"
    torch.save({"X": X_new, "y": gold_sents_new, "z": z_new, "masks": masks_new, "true_length": true_length_new, "labels": y_new},
               f"../data/biosbias/LM_tokens_{type}_{sampling}_roberta-base_128_{split_type}-seed_{seed}.pt")
    torch.save({"X": X2_new, "y": gold_sents2_new, "z": z2_new, "masks": masks2_new, "true_length": true_length2_new, "labels": y2_new},
               f"../data/biosbias/LM_tokens_{type}_{sampling}_roberta-base_128_{split_type}_mask_length_2-seed_{seed}.pt")
    torch.save({"X": X3_new, "y": gold_sents3_new, "z": z3_new, "masks": masks3_new, "true_length": true_length3_new, "labels": y3_new},
               f"../data/biosbias/LM_tokens_{type}_{sampling}_roberta-base_128_{split_type}_mask_length_3-seed_{seed}.pt")
    torch.save({"X": X4_new, "y": gold_sents4_new, "z": z4_new, "masks": masks4_new, "true_length": true_length4_new, "labels": y4_new},
               f"../data/biosbias/LM_tokens_{type}_{sampling}_roberta-base_128_{split_type}_mask_length_4-seed_{seed}.pt")

def balance_dataset(X, y, z, masks=None, other=None, oversampling=False):
    indexes = []

    for label in np.unique(y):
        female_idx_bool = np.logical_and(y == label, z == 'F')
        male_idx_bool = np.logical_and(y == label, z == 'M')
        female_idx = np.arange(len(y))[female_idx_bool]
        male_idx = np.arange(len(y))[male_idx_bool]

        n_female = np.sum(female_idx_bool)
        n_male = np.sum(male_idx_bool)

        if oversampling:
            size = max(n_female, n_male)
            if n_female > n_male:
                sampled_female_idx = np.random.choice(female_idx, size=size, replace=False)
                sampled_male_idx = np.random.choice(male_idx, size=size, replace=True)
            else:
                sampled_female_idx = np.random.choice(female_idx, size=size, replace=True)
                sampled_male_idx = np.random.choice(male_idx, size=size, replace=False)

        else:
            size = min(n_female, n_male)
            sampled_female_idx = np.random.choice(female_idx, size=size, replace=False)
            sampled_male_idx = np.random.choice(male_idx, size=size, replace=False)

        indexes += sampled_female_idx.tolist()
        indexes += sampled_male_idx.tolist()

    other_output = []
    if other is not None:
        for t in other:
            other_output.append(t[indexes])
        if masks is not None:
            return (X[indexes], y[indexes], z[indexes], masks[indexes], *other_output)
        else:
            return (X[indexes], y[indexes], z[indexes], *other_output)
    else:
        if masks is not None:
            return X[indexes], y[indexes], z[indexes], masks[indexes]
        else:
            return X[indexes], y[indexes], z[indexes]

def BiasinBios_create_percentage_data(split_type, seed):
    data = torch.load(f"../data/biosbias/tokens_raw_original_roberta-base_128_{split_type}-seed_{seed}.pt")
    golden_y = data['y']
    z = data['z']
    perc = {}

    for profession in np.unique(golden_y):
        total_of_label = len(golden_y[golden_y == profession])
        indices_female = np.logical_and(golden_y == profession, z == 'F')
        perc_female = len(golden_y[indices_female]) / total_of_label
        perc[profession] = perc_female

    torch.save(perc, f"../data/biosbias/perc_{split_type}-seed_{seed}")

def BiasInBios_extract_tokens_data(datatype, tokenizer, model_name):
    f = open('../data/biosbias/BIOS.pkl', 'rb')
    ds = pickle.load(f)

    labels = []
    genders = []
    inputs = []

    for r in tqdm(ds):
        if datatype == "name":
            sent = " ".join(r['name'])
        elif datatype == "scrubbed":
            sent = r["bio"]  # no start_pos needed
        else:
            sent = r["raw"][r["start_pos"]:]

        inputs.append(sent)
        labels.append(r["title"])
        genders.append(r["gender"])

    if datatype == "name":
        encoded_dict = tokenizer(inputs, add_special_tokens=True, padding=True, return_attention_mask=True)
    else:
        max_length = 128
        encoded_dict = tokenizer(inputs, add_special_tokens=True, padding='max_length', max_length=max_length, truncation=True, return_attention_mask=True)

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    # Convert the lists into numpy arrays.
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    genders = np.array(genders)
    torch.save({"X": input_ids, "masks": attention_masks, "y": labels, "z": genders}, f"../data/biosbias/tokens_{datatype}_{model_name}_128.pt")

# def BiasInBios_extract_vectors_data(type, model, tokenizer,folder = "../data/biosbias", data_path='../data/biosbias/BIOS.pkl'):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     max_length = 128
#     f = open(data_path, 'rb')
#     ds = pickle.load(f)
#     f.close()
#     vectors = []
#     labels = []
#     genders = []
#     indices = []
#     # ds = ds[:10000]
#
#     with torch.no_grad():
#         model.eval()
#         for i, r in enumerate(tqdm(ds)):
#             if type == "scrubbed":
#                 sent = r["bio"]  # no start_pos needed
#             elif type == "name":
#                 sent = " ".join(r['name'])
#             else:
#                 sent = r["raw"][r["start_pos"]:]
#
#             input_ids = torch.tensor(tokenizer.encode(
#                 sent,
#                 add_special_tokens=True,
#                 max_length=max_length,
#                 truncation=True)).unsqueeze(0).to(device)
#
#             if type == "name":
#                 v = model.embeddings(input_ids)[0].mean(dim=0).cpu().detach().numpy()
#             else:
#                 v = model(input_ids).last_hidden_state[:, 0, :][0].cpu().detach().numpy()
#
#             vectors.append(v)
#             labels.append(r["title"])
#             genders.append(r["gender"])
#             indices.append(i)
#
#         vectors = np.array(vectors)
#         labels = np.array(labels)
#         genders = np.array(genders)
#
#     Path(folder).mkdir(parents=True, exist_ok=True)
#     torch.save({"X": vectors, "y": labels, "z": genders, "original index": indices}, folder + f"/vectors_{type}_roberta-base_128.pt")


def BiasInBios_extract_vectors_data(type, model, data_path, feature_extractor='roberta-base', folder = "../data/biosbias"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(data_path)

    vectors = []
    labels = []
    genders = []

    X = torch.tensor(data['X']).to(device)
    y = data['y']
    z = data['z']
    masks = torch.tensor(data['masks']).to(device)

    with torch.no_grad():
        model.eval()

        for i, x in enumerate(tqdm(X)):
            input_ids = x
            v = model(input_ids.unsqueeze(0), attention_mask=masks[i].unsqueeze(0)).last_hidden_state[:, 0, :][0].cpu().detach().numpy()

            vectors.append(v)
            labels.append(y[i])
            genders.append(z[i])

        vectors = np.array(vectors)
        labels = np.array(labels)
        genders = np.array(genders)

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save({"X": vectors, "y": labels, "z": genders}, folder + f"/vectors_{type}_{feature_extractor}_128.pt")


def convert_bias_in_bios_to_LM(biography, tokenizer, mask_len=None, anonymize=False):
    if biography["title"][0] in ('a', 'e', 'i', 'o', 'u'):
        article = "an"
    else:
        article = "a"
    title = biography["title"].replace("_", " ")

    tokens = tokenizer(f" {article} {title}", add_special_tokens = False)['input_ids'] # " a dietitian"

    if mask_len is None:
        mask_len = (len(tokens))

    masks = "<mask> " * (mask_len-1) + "<mask>"

    if anonymize:
        name_replacement = "_"
        return {
            "X": f"{name_replacement} is {masks}." + biography["bio"].replace(title, "_"),
            "y": f"{name_replacement} is {article} {title}." + biography["bio"].replace(title, "_"),
            "true_length": len(tokens)
        }

    else:
        an = re.search(" is an ", biography["raw"])
        a = re.search(" is a ", biography["raw"])
        if a:
            beginning = a.span()[0]
            if an:
                beginning = an.span()[0] if an.span()[0] < a.span()[0] else a.span()[0]
        else:  # there will always we an or a so no need to verify
            beginning = an.span()[0]

        to_replace = biography["raw"][beginning:biography["start_pos"]]
        prefix = biography['raw'][:biography["start_pos"]]
        suffix = biography['raw'][biography["start_pos"]:].replace(title, "_") # why replace?
        return {
            "X": prefix.replace(to_replace, f" is {masks}.") + suffix,
            "y": prefix.replace(to_replace, f" is {article} {title}.") + suffix,
            "true_length": len(tokens)
        }


def create_biasinbios_LM_dataset_aux(data, datatype, seed, data_type='raw', mask_len=None):
    model_version = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_version)

    max_length = 128
    sents = []
    gold_sents = []
    genders = []
    masks = []
    lengths = []
    labels = []
    for biography in tqdm(data):
        conversion = convert_bias_in_bios_to_LM(biography, tokenizer, mask_len=mask_len,
                                                anonymize=True if data_type == 'scrubbed' else False)
        X, y, true_length = conversion['X'], conversion['y'], conversion["true_length"]
        X_encoded = tokenizer(
            X,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
        )
        y_encoded = tokenizer(
            y,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
        )

        sents.append(X_encoded['input_ids'])
        gold_sents.append(y_encoded['input_ids'])
        genders.append(biography['gender'])
        masks.append(X_encoded['attention_mask'])
        lengths.append(true_length)
        labels.append(biography['title'])

    sents = np.array(sents)
    gold_sents = np.array(gold_sents)
    genders = np.array(genders)
    masks = np.array(masks)
    lengths = np.array(lengths)
    labels = np.array(labels)

    if mask_len:
        torch.save(
            {"X": sents, "masks": masks, "y": gold_sents, "z": genders, "true_length": lengths, "labels": labels},
            f"../data/biosbias/LM_tokens_{data_type}_original_roberta-base_128_{datatype}_mask_length_{mask_len}-seed_{seed}.pt")
    else:
        torch.save(
            {"X": sents, "masks": masks, "y": gold_sents, "z": genders, "true_length": lengths, "labels": labels},
            f"../data/biosbias/LM_tokens_{data_type}_original_roberta-base_128_{datatype}-seed_{seed}.pt")


def create_biasinbios_LM_dataset(path, seed, data_type='raw',mask_len=None):
    import sys
    sys.path.insert(0, '..')

    with open(path, 'rb') as f:
        data = pickle.load(f)

    y = [r["title"] for r in data]
    data_train_valid, data_test = train_test_split(data, random_state=seed, stratify=y, test_size=0.25)
    y_train_valid = [r["title"] for r in data_train_valid]
    data_train, data_valid = train_test_split(data_train_valid, random_state=seed, stratify=y_train_valid,
                                              test_size=0.133)
    print(f"train length: {len(data_train)}, valid length: {len(data_valid)}, test length: {len(data_test)}")

    create_biasinbios_LM_dataset_aux(data_train, "train", seed, data_type, mask_len)
    create_biasinbios_LM_dataset_aux(data_test, "test", seed, data_type, mask_len)
    create_biasinbios_LM_dataset_aux(data_valid, "valid", seed, data_type, mask_len)