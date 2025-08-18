from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image captions using BLIP model")
    parser.add_argument("--data_path", type=str, default="/mimer/NOBACKUP/groups/eeg_foundation_models/NeuroLM/data/things_eeg_2",
                        help="Path to the dataset directory containing images")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the generated captions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_path = data_path if args.save_path is None else args.save_path
    img_parent_dir  = os.path.join(data_path, 'images')
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    train_captions = {}
    print("Generating captions for training images...")
    for item in range(len(img_metadata['train_img_files'])):
        abs_img_path = os.path.join(img_parent_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])

        img = Image.open(abs_img_path).convert('RGB')
        inputs = processor(img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        train_captions[img_metadata['train_img_files'][item]] = caption

        if item % 500 == 0:
            print(f"Processed {item} training images...")

    test_captions = {}
    print("Generating captions for test images...")
    for item in range(len(img_metadata['test_img_files'])):
        abs_img_path = os.path.join(img_parent_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])

        img = Image.open(abs_img_path).convert('RGB')
        inputs = processor(img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        test_captions[img_metadata['test_img_files'][item]] = caption

    # Save dictionaries
    np.save(os.path.join(save_path, 'train_captions.npy'), train_captions, allow_pickle=True)
    np.save(os.path.join(save_path, 'test_captions.npy'), test_captions, allow_pickle=True)

    print("Caption generation complete.")
