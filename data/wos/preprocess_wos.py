import json
import re
from sklearn.model_selection import train_test_split

"""
WoS Reference: https://github.com/kk7nc/HDLTex
"""

FILE_DIR = "Data.txt"
OUTPUT_FILE = "wos.jsonl"

stats = {
    "Root": {
        "CS": 0,
        "Medical": 0,
        "Civil": 0,
        "ECE": 0,
        "biochemistry": 0,
        "MAE": 0,
        "Psychology": 0,
    },
    "CS": {
        "Symbolic computation": 402,
        "Computer vision": 432,
        "Computer graphics": 412,
        "Operating systems": 380,
        "Machine learning": 398,
        "Data structures": 392,
        "network security": 445,
        "Image processing": 415,
        "Parallel computing": 443,
        "Distributed computing": 403,
        "Algorithm design": 379,
        "Computer programming": 425,
        "Relational databases": 377,
        "Software engineering": 416,
        "Bioinformatics": 365,
        "Cryptography": 387,
        "Structured Storage": 43,
    },
    "Medical": {
        "Alzheimer's Disease": 368,
        "Parkinson's Disease": 298,
        "Sprains and Strains": 142,
        "Cancer": 359,
        "Sports Injuries": 365,
        "Senior Health": 118,
        "Multiple Sclerosis": 253,
        "Hepatitis C": 288,
        "Weight Loss": 327,
        "Low Testosterone": 305,
        "Fungal Infection": 372,
        "Diabetes": 353,
        "Parenting": 343,
        "Birth Control": 335,
        "Heart Disease": 291,
        "Allergies": 357,
        "Menopause": 371,
        "Emergency Contraception": 291,
        "Skin Care": 339,
        "Myelofibrosis": 198,
        "Hypothyroidism": 315,
        "Headache": 341,
        "Overactive Bladder": 340,
        "Irritable Bowel Syndrome": 336,
        "Polycythemia Vera": 148,
        "Atrial Fibrillation": 294,
        "Smoking Cessation": 257,
        "Lymphoma": 267,
        "Asthma": 317,
        "Bipolar Disorder": 260,
        "Crohn's Disease": 198,
        "Idiopathic Pulmonary Fibrosis": 246,
        "Mental Health": 222,
        "Dementia": 237,
        "Rheumatoid Arthritis": 188,
        "Osteoporosis": 320,
        "Medicare": 255,
        "Psoriatic Arthritis": 202,
        "Addiction": 309,
        "Atopic Dermatitis": 262,
        "Digestive Health": 95,
        "Healthy Sleep": 129,
        "Anxiety": 262,
        "Psoriasis": 128,
        "Ankylosing Spondylitis": 321,
        "Children's Health": 350,
        "Stress Management": 361,
        "HIV/AIDS": 358,
        "Depression": 130,
        "Migraine": 178,
        "Osteoarthritis": 305,
        "Hereditary Angioedema": 182,
        "Kidney Health": 90,
        "Autism": 309,
        "Schizophrenia": 38,
        "Outdoor Health": 2,
    },
    "Civil": {
        "Green Building": 418,
        "Water Pollution": 446,
        "Smart Material": 363,
        "Ambient Intelligence": 410,
        "Construction Management": 412,
        "Suspension Bridge": 395,
        "Geotextile": 419,
        "Stealth Technology": 148,
        "Solar Energy": 384,
        "Remote Sensing": 384,
        "Rainwater Harvesting": 441,
        "Transparent Concrete": 3,
        "Highway Network System": 4,
        "Nano Concrete": 7,
        "Bamboo as a Building Material": 2,
        "Underwater Windmill": 1,
    },
    "ECE": {
        "Electric motor": 372,
        "Satellite radio": 148,
        "Digital control": 426,
        "Microcontroller": 413,
        "Electrical network": 392,
        "Electrical generator": 240,
        "Electricity": 447,
        "Operational amplifier": 419,
        "Analog signal processing": 407,
        "State space representation": 344,
        "Signal-flow graph": 274,
        "Electrical circuits": 375,
        "Lorentz force law": 44,
        "System identification": 417,
        "PID controller": 429,
        "Voltage law": 54,
        "Control engineering": 276,
        "Single-phase electric power": 6,
    },
    "biochemistry": {
        "Molecular biology": 746,
        "Enzymology": 576,
        "Southern blotting": 510,
        "Northern blotting": 699,
        "Human Metabolism": 622,
        "Polymerase chain reaction": 750,
        "Immunology": 652,
        "Genetics": 566,
        "Cell biology": 552,
        "DNA/RNA sequencing": 14,
    },
    "MAE": {
        "Fluid mechanics": 386,
        "Hydraulics": 402,
        "computer-aided design": 371,
        "Manufacturing engineering": 346,
        "Machine design": 420,
        "Thermodynamics": 361,
        "Materials Engineering": 289,
        "Strength of materials": 335,
        "Internal combustion engine": 387,
    },
    "Psychology": {
        "Prenatal development": 389,
        "Attention": 416,
        "Eating disorders": 387,
        "Borderline personality disorder": 376,
        "Prosocial behavior": 388,
        "False memories": 362,
        "Problem-solving": 360,
        "Prejudice": 389,
        "Antisocial personality disorder": 368,
        "Nonverbal communication": 394,
        "Leadership": 350,
        "Child abuse": 404,
        "Gender roles": 395,
        "Depression": 380,
        "Social cognition": 397,
        "Seasonal affective disorder": 365,
        "Person perception": 391,
        "Media violence": 296,
        "Schizophrenia": 335,
    },
}

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def process_wos_data():
    """
    Main function to process Web of Science data.
    Reads the input file in UTF-16 encoding, processes each document,
    and writes the results to a JSONL file in UTF-8 encoding.

    Output format:
    {
        "text": "processed document text",
        "label": ["level1_label", "level2_label"]
    }
    """
    # Read input file in UTF-16 encoding
    with open(FILE_DIR, "r", encoding="utf-8") as f:
        origin_txt = f.readlines()

    # Dictionary to track multiple labels for same ID
    label_check = {}
    output_data = []

    # Process each line (skip header)
    for line in origin_txt[1:]:
        line = line.rstrip("\n")
        line = line.split("\t")
        assert len(line) == 7

        # Extract and clean labels
        sample_label = [line[3].strip(), line[4].strip()]
        code = str(line[0]) + "-" + str(line[1])

        # Handle multiple labels for same ID
        if code in label_check:
            if sample_label[1] not in label_check[code]:
                label_check[code].append(sample_label[1])
        else:
            label_check[code] = [sample_label[1]]

        # Choose the label with highest sample count
        for i in label_check[code]:
            if stats[sample_label[0]][i] > stats[sample_label[0]][sample_label[1]]:
                sample_label[1] = i
                break

        cleaned_text = clean_text(line[6])
        # Add to output data
        output_data.append(
            {"text": cleaned_text, "label": [label.lower() for label in sample_label]}
        )

    # Print statistics about multiple labels
    multiple_labels = 0
    for code, labels in label_check.items():
        if len(labels) > 1:
            print(f"ID {code} has multiple labels: {labels}")
            multiple_labels += len(labels) - 1

    print(
        f"Total IDs with multiple labels: {sum(1 for labels in label_check.values() if len(labels) > 1)}"
    )
    print(f"Total extra labels: {multiple_labels}")
    print(f"Total unique IDs: {len(label_check)}")

    # Write to JSONL file in UTF-8 encoding
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in output_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Successfully processed {len(output_data)} documents")
    print(f"Output saved to {OUTPUT_FILE}")
    # Split data into train, validation, and test sets with specified sizes
    test_size = 0.2
    val_size = 0.2
    random_state = 42
    # First split off test set
    train_val_data, test_data = train_test_split(output_data, test_size=test_size, random_state=random_state)
    # Then split train_val_data into train and validation sets
    val_ratio = val_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=random_state)
    # Write split data to JSONL files
    with open("wos_train.jsonl", "w", encoding="utf-8") as f_train:
        for item in train_data:
            json.dump(item, f_train, ensure_ascii=False)
            f_train.write("\n")
    with open("wos_val.jsonl", "w", encoding="utf-8") as f_val:
        for item in val_data:
            json.dump(item, f_val, ensure_ascii=False)
            f_val.write("\n")
    with open("wos_test.jsonl", "w", encoding="utf-8") as f_test:
        for item in test_data:
            json.dump(item, f_test, ensure_ascii=False)
            f_test.write("\n")
    print(f"Data split completed: train {len(train_data)}, val {len(val_data)}, test {len(test_data)}")



if __name__ == "__main__":
    process_wos_data()
