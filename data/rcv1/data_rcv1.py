import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.dom.minidom
import pandas as pd
import json
import csv
from collections import defaultdict

np.random.seed(7)


def main():
    source = []
    labels = []
    titles = []
    dates = []
    label_dict = {}
    hiera = defaultdict(set)

    with open("rcv1.taxonomy", "r") as f:
        label_dict["Root"] = -1
        for line in f.readlines():
            line = line.strip().split("\t")
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop("Root")
        hiera.pop(-1)

    data = pd.read_csv("rcv1_v2.csv")
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line["text"])
        root = dom.documentElement

        title_tags = root.getElementsByTagName("title")
        title = title_tags[0].firstChild.data if title_tags else ""

        date = root.getAttribute("date")

        tags = root.getElementsByTagName("p")
        text = ""
        for tag in tags:
            text += tag.firstChild.data
        if text == "":
            continue

        source.append(text)
        titles.append(title)
        dates.append(date)
        l = line["label"].split("'")
        labels.append([label_dict[i] for i in l[1::2]])
    print(len(labels))

    data = pd.read_csv("rcv1_v2.csv")
    ids = []
    valid_indices = []
    idx = 0
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line["text"])
        root = dom.documentElement
        tags = root.getElementsByTagName("p")
        cont = False
        for tag in tags:
            if tag.firstChild.data != "":
                cont = True
                break
        if cont:
            ids.append(line["id"])
            valid_indices.append(idx)
            idx += 1

    train_ids = []
    with open("lyrl2004_tokens_train.dat", "r") as f:
        for line in f.readlines():
            if line.startswith(".I"):
                train_ids.append(int(line[3:-1]))

    train_ids = set(train_ids)
    train = []
    test = []
    for i in range(len(ids)):
        if ids[i] in train_ids:
            train.append(valid_indices[i])
        else:
            test.append(valid_indices[i])

    id = [i for i in range(len(train))]
    np_data = np.array(train)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, val = train_test_split(train, test_size=0.1, random_state=0)

    inv_label = {i: v for v, i in label_dict.items()}

    topic_mappings = {}
    with open("rcv1_v2_topics_desc.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_mappings[row["topic_code"]] = row["topic_desc"].lower()

    with open("rcv1_train.jsonl", "w", encoding="utf-8") as f:
        for i in train:
            record = {
                "text": source[i],
                "label": [
                    topic_mappings.get(inv_label[l], inv_label[l]) for l in labels[i]
                ],
                "title": titles[i],
                "date": dates[i],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with open("rcv1_val.jsonl", "w", encoding="utf-8") as f:
        for i in val:
            record = {
                "text": source[i],
                "label": [
                    topic_mappings.get(inv_label[l], inv_label[l]) for l in labels[i]
                ],
                "title": titles[i],
                "date": dates[i],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with open("rcv1_test.jsonl", "w", encoding="utf-8") as f:
        for i in test:
            record = {
                "text": source[i],
                "label": [
                    topic_mappings.get(inv_label[l], inv_label[l]) for l in labels[i]
                ],
                "title": titles[i],
                "date": dates[i],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
