import os
from datasets import load_dataset
from PIL import Image
import json

splits = ["train", "test"]
output_parent_folder = "brain-vqa-rad"
os.makedirs(output_parent_folder, exist_ok=True)


def filter_infarct(example):
    return "brain" in example["question"].lower()

def convert_and_save(sample, idx, image_output_folder):    
    sample_id = f"{idx:012d}"
    image_filename = f"{sample_id}.jpg"
    image_path = os.path.join(image_output_folder, image_filename)
    if isinstance(sample["image"], Image.Image):
        sample["image"].save(image_path, format="JPEG")
    else:
        raise ValueError("sample['image'] is not a PIL Image object")
    
    llava_entry = {
        "id": sample_id,
        "image": image_filename,  
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{sample['question']}"
            },
            {
                "from": "gpt",
                "value": sample["answer"]
            }
        ]
    }
    return llava_entry

for split in splits:
    print(f"Processing split: {split}")
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
    except Exception as e:
        print(f"Error loading split '{split}': {e}")
        continue

    filtered_dataset = dataset.filter(filter_infarct)
    print(f"Filtered {len(filtered_dataset)} samples in split '{split}'.")

    image_output_folder = os.path.join(output_parent_folder, f"{split}_images")
    os.makedirs(image_output_folder, exist_ok=True)

    llava_entries = []
    for idx, sample in enumerate(filtered_dataset):
        try:
            llava_entry = convert_and_save(sample, idx, image_output_folder)
            llava_entries.append(llava_entry)
        except Exception as e:
            print(f"Error processing sample idx {idx} in split '{split}': {e}")

    
    output_json_path = os.path.join(output_parent_folder, f"{split}_vqa_rad_llava.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(llava_entries, f, indent=2, ensure_ascii=False)

    print(f"Split '{split}': {len(llava_entries)} samples converted.")
    print(f"JSON file saved to: {output_json_path}")
    print(f"Images saved to folder: {os.path.abspath(image_output_folder)}\n")
