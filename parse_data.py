import argparse
import json
import os
import pickle

import clip
import pandas as pd
import skimage.io as io
import torch
from clip.model import CLIP
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms.transforms import Compose
from tqdm import tqdm

SEED = 333
DEVICE = "cuda:0"
TORCH_DEVICE = torch.device(DEVICE)
DATA_PATH = "./deepfashion-mm/"
CAPT_PATH = os.path.join(DATA_PATH, "captions.json")


class StratifiedSampling:
    def __init__(self, data: dict):
        self.keys_list = list(data.keys())


    def create_df_from_keys(self) -> pd.Series:
        file_array = pd.Series(self.keys_list)
        df = file_array.str.split(pat="-", expand=True)
        df.columns = ['department', 'group', 'id', 'type']
        df['filename'] = file_array

        return df


    def get_samples(self) -> dict:
        df_files = self.create_df_from_keys()
        
        train_keys, valtest_keys = train_test_split(
            self.keys_list, 
            test_size=0.15, 
            stratify=df_files[["department", "group"]], 
            random_state=SEED
        )

        valtest_cond = df_files['filename'].isin(valtest_keys)

        val_keys, test_keys = train_test_split(
            valtest_keys,
            test_size=0.5, 
            stratify=df_files.loc[valtest_cond, ["department", "group"]], 
            random_state=SEED
        )

        return {"train": train_keys, "val": val_keys, "test": test_keys}


class ImageEncoder:
    def __init__(self, clip_model_type: str):
        self.clip_model_type = clip_model_type


    def fit(self, data: dict):
        clip_model, preprocess = clip.load(self.clip_model_type, device=TORCH_DEVICE, jit=False)
        clip_model_name = self.clip_model_type.replace("/", "_")
        sampler = StratifiedSampling(data)
        data_splits = sampler.get_samples()

        for split_name, split_data in data_splits.items():
            self.encode_data(
                split_name, 
                split_data,
                data,
                clip_model,
                clip_model_name,
                preprocess
            )

        return data_splits


    @staticmethod
    def encode_data(
        split_name: str, 
        split_data: list, 
        all_data: dict, 
        clip_model: CLIP,
        clip_model_name: str, 
        preprocess: Compose
        ) -> None:
        
        out_path = os.path.join(DATA_PATH, f"deepfashion_{clip_model_name}_{split_name}.pkl")
        all_embeddings = []
        all_captions = []
        
        for i, filename in tqdm(enumerate(split_data)):
            d = dict()
            d["image_id"] = filename
            d["caption"] = all_data[filename]
            d["clip_embedding"] = i
            filepath = os.path.join(DATA_PATH, f"images/{filename}")

            image = io.imread(filepath)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(TORCH_DEVICE)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            all_embeddings.append(prefix)
            all_captions.append(d)
        
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "captions": all_captions,
                },
                f,
            )

        print(f"{len(all_embeddings)} {split_name} split embeddings saved at {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()

    with open(CAPT_PATH, "r") as f:
        data = json.load(f)
    print(f"{len(data)} captions loaded from json.")

    encoder = ImageEncoder(args.clip_model_type)
    
    data_splits = encoder.fit(data)
    clip_model_name = args.clip_model_type.replace("/", "_")
    split_path = os.path.join(DATA_PATH, f"data_splits_{clip_model_name}.json")

    with open(split_path, "w") as f:
        json.dump(data_splits, f)
    print(f"Train and val splits saved into {split_path}")
    