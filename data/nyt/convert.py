import json

input_file = "nyt_val_all_new.json"
output_file = "val_da.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(
    output_file, "w", encoding="utf-8"
) as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
            continue

        # 处理 label 字段，转换为如下形式:
        # ["News", "New York and Region", "Connecticut", "Opinion", "Opinion"]
        labels = record.get("label", [])
        processed_labels = []
        for label in labels:
            # 如果以 "Top/" 开头，则删除该前缀
            if label.startswith("Top/"):
                label = label[len("Top/") :]
            # 按照 "/" 分割，并取最后一个部分
            last_component = label.split("/")[-1]
            processed_labels.append(last_component)

        record["label"] = processed_labels

        # 将处理后的 JSON 对象写入到新的文件
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
