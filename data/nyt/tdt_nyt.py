from concurrent.futures import ThreadPoolExecutor
import os
import json
from pathlib import Path
from xml.etree import ElementTree
from datetime import datetime
from tqdm import tqdm
from threading import Lock
from collections import defaultdict


class XMLProcessor:
    def __init__(self, output_file, max_workers=8):
        self.output_file = output_file
        self.max_workers = max_workers
        self.results = defaultdict(list)
        self.lock = Lock()

    def get_sorted_xml_files(self, root_dir):
        xml_files = []
        for dirpath, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".xml"):
                    full_path = os.path.join(dirpath, file)
                    parts = Path(full_path).parts[-4:-1]
                    xml_files.append((full_path, parts))

        xml_files.sort(key=lambda x: x[1])
        return [f[0] for f in xml_files]

    def parse_xml_file(self, file_path):
        try:
            tree = ElementTree.parse(file_path)
            root = tree.getroot()

            title = root.find(".//title")
            title = title.text if title is not None else ""

            paragraphs = root.findall(".//p")
            text = " ".join(p.text for p in paragraphs if p.text)

            path_parts = Path(file_path).parts
            date = f"{path_parts[-4]}/{path_parts[-3]}/{path_parts[-2]}"

            with self.lock:
                self.results[path_parts[-4]].append(
                    {
                        "date": date,
                        "title": title,
                        "text": text,
                    }
                )

            return True
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {str(e)}")
            return False

    def write_results(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            for year in sorted(self.results.keys()):
                for item in self.results[year]:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

    def process_files(self, input_dir):
        xml_files = self.get_sorted_xml_files(input_dir)
        total_files = len(xml_files)
        print(f"Found {total_files} XML files")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(
                tqdm(
                    executor.map(self.parse_xml_file, xml_files),
                    total=total_files,
                    desc="Processing progress",
                )
            )

        self.write_results()


def main():
    input_dir = "Nytimes"
    output_file = "nytimes.jsonl"

    start_time = datetime.now()
    print(f"Start processing, time: {start_time}")

    processor = XMLProcessor(output_file, max_workers=8)
    processor.process_files(input_dir)

    end_time = datetime.now()
    print(f"Processing completed, time: {end_time}")
    print(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
