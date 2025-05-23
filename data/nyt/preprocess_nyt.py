#!/usr/bin/env python3
import os
import xml.dom.minidom
from tqdm import tqdm
import json
import re
import tarfile
import shutil
from collections import defaultdict
import torch

"""
NYTimes Reference: https://catalog.ldc.upenn.edu/LDC2008T19
"""

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 200

source = []
labels = []
label_ids = []
label_dict = {}
sentence_ids = []
hiera = defaultdict(set)

dates = []
titles = []

ROOT_DIR = "Nytimes/"
label_f = "nyt_label.vocab"


def read_nyt(id_json):
    with open(id_json, "r") as f:
        ids = f.readlines()
    print(ids[:2])
    with open(label_f, "r") as f:
        label_vocab_s = f.readlines()
    label_vocab = [label.strip() for label in label_vocab_s]

    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus = []

    for file_name in tqdm(ids):
        xml_path = file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement

            title_str = ""
            title_tags = root.getElementsByTagName("title")
            if title_tags and title_tags[0].firstChild is not None:
                title_str = title_tags[0].firstChild.data.strip()
            else:
                hl1_tags = root.getElementsByTagName("hl1")
                if hl1_tags and hl1_tags[0].firstChild is not None:
                    title_str = hl1_tags[0].firstChild.data.strip()

            date_str = ""
            date_tags = root.getElementsByTagName("date")
            if date_tags and date_tags[0].firstChild is not None:
                date_str = date_tags[0].firstChild.data.strip()
                if re.fullmatch(r"\d{8}", date_str):
                    date_str = date_str[:4] + "/" + date_str[4:6] + "/" + date_str[6:8]
            else:
                pubdata_tags = root.getElementsByTagName("pubdata")
                if pubdata_tags:
                    pub_date = pubdata_tags[0].getAttribute("date.publication")
                    if pub_date and len(pub_date) >= 8 and pub_date[:8].isdigit():
                        date_str = (
                            pub_date[:4] + "/" + pub_date[4:6] + "/" + pub_date[6:8]
                        )

            p_tags = root.getElementsByTagName("p")
            text = ""
            for tag in p_tags[1:]:
                if tag.firstChild is not None:
                    text += tag.firstChild.data
            if text == "":
                continue
            source.append(text)
            titles.append(title_str)
            dates.append(date_str)

            sample_label = []
            classifier_tags = root.getElementsByTagName("classifier")
            for tag in classifier_tags:
                type_val = tag.getAttribute("type")
                if type_val != "taxonomic_classifier":
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split("/")
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = "/".join(hier_list[:l])
                    if label == "Top":
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
            labels.append(sample_label)
            sentence_ids.append(file_name)
            sample["doc_topic"] = []
            sample["doc_keyword"] = []
            corpus.append(sample)
        except Exception as e:
            print(xml_path)
            print("Something went wrong...", e)
            continue


if __name__ == "__main__":
    files = os.listdir("nyt_corpus/data")
    for year in files:
        month = os.listdir(os.path.join("nyt_corpus/data", year))
        for m in month:
            f_tar = tarfile.open(os.path.join("nyt_corpus/data", year, m))
            f_tar.extractall(os.path.join("Nytimes", year))
    files = os.listdir("Nytimes")
    for year in files:
        month = os.listdir(os.path.join("Nytimes", year))
        for m in month:
            days = os.listdir(os.path.join("Nytimes", year, m))
            for d in days:
                file_list = os.listdir(os.path.join("Nytimes", year, m, d))
                for f_name in file_list:
                    shutil.move(
                        os.path.join("Nytimes", year, m, d, f_name),
                        os.path.join("Nytimes", year, f_name),
                    )
    read_nyt("idnewnyt_train.json")
    read_nyt("idnewnyt_val.json")
    read_nyt("idnewnyt_test.json")
    rev_dict = {}
    for l in labels:
        for l_ in l:
            split = l_.split("/")
            if l_ not in label_dict:
                label_dict[l_] = len(label_dict)
            for i in range(1, len(split) - 1):
                hiera[label_dict["/".join(split[: i + 1])]].add(
                    label_dict["/".join(split[: i + 2])]
                )
                assert "/".join(split[: i + 2]) not in rev_dict or rev_dict[
                    "/".join(split[: i + 2])
                ] == "/".join(split[: i + 1])
                rev_dict["/".join(split[: i + 2])] = "/".join(split[: i + 1])
    for l in labels:
        one_hot = [0] * len(label_dict)
        for i in l:
            one_hot[label_dict[i]] = 1
        label_ids.append(one_hot)

    train_split = open("idnewnyt_train.json", "r").readlines()
    dev_split = open("idnewnyt_val.json", "r").readlines()
    test_split = open("idnewnyt_test.json", "r").readlines()
    train, test, val = [], [], []
    for t in train_split:
        train.append(sentence_ids.index(t))
    for t in dev_split:
        val.append(sentence_ids.index(t))
    for t in test_split:
        test.append(sentence_ids.index(t))

    with open("nyt_train.json", "w") as f_out:
        for i in train:
            raw_labels = labels[i]
            processed_labels = []
            for lb in raw_labels:
                if lb.startswith("Top/"):
                    lb = lb[len("Top/") :]
                lb_last = lb.split("/")[-1]
                if lb_last not in processed_labels:
                    processed_labels.append(lb_last)
            line = json.dumps(
                {
                    "date": dates[i],
                    "title": titles[i],
                    "label": processed_labels,
                    "text": source[i],
                }
            )
            f_out.write(line + "\n")
    with open("nyt_val.json", "w") as f_out:
        for i in val:
            raw_labels = labels[i]
            processed_labels = []
            for lb in raw_labels:
                if lb.startswith("Top/"):
                    lb = lb[len("Top/") :]
                lb_last = lb.split("/")[-1]
                if lb_last not in processed_labels:
                    processed_labels.append(lb_last)
            line = json.dumps(
                {
                    "date": dates[i],
                    "title": titles[i],
                    "label": processed_labels,
                    "text": source[i],
                }
            )
            f_out.write(line + "\n")
    with open("nyt_test.json", "w") as f_out:
        for i in test:
            raw_labels = labels[i]
            processed_labels = []
            for lb in raw_labels:
                if lb.startswith("Top/"):
                    lb = lb[len("Top/") :]
                lb_last = lb.split("/")[-1]
                if lb_last not in processed_labels:
                    processed_labels.append(lb_last)
            line = json.dumps(
                {
                    "date": dates[i],
                    "title": titles[i],
                    "label": processed_labels,
                    "text": source[i],
                }
            )
            f_out.write(line + "\n")
