import argparse
import json
import os
from typing import Any

import clip
import numpy as np
import PIL
import skimage.io as io
import torch
import torch.nn.functional as nnf
from tqdm import tqdm
from transformers import GPT2Tokenizer

from train import MLP, ClipCaptionModel, ClipCaptionPrefix

DEVICE = torch.device("cuda:0")
CPU = torch.device("cpu")


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=DEVICE)
    is_stopped = torch.zeros(beam_size, device=DEVICE, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(DEVICE)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(DEVICE)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class Predictor:
    def __init__(self, model: Any, args: argparse.Namespace):
        """Load the model into memory to make running multiple predictions efficient"""
        self.clip_model, self.preprocess = clip.load(
            args.clip_model_type, DEVICE, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = model
        self.model_name = args.model_name
        self.prefix_length = args.prefix_length
        self.use_beam_search = args.use_beam_search
        self.img_path = args.image_path
        self.data_path = args.data_path

    def single_predict(self, image):
        """Run a single prediction on the model"""
        image = io.imread(image)
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(DEVICE, dtype=torch.float32)
            prefix_embed = self.model.clip_project(prefix).reshape(
                1, self.prefix_length, -1
            )
        if self.use_beam_search:
            return generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(self.model, self.tokenizer, embed=prefix_embed)

    def predict(self):
        """Run predict for a dataset"""

        with open(self.data_path, "r") as f:
            data = json.load(f)
        print(f"{len(data)} captions loaded from json.")

        test_data = data["test"]
        res_captions = []

        test_progress = tqdm(total=len(data["test"]), desc=self.model_name)
        for i in range(len(data["test"])):
            img_id = test_data[i]
            filename = os.path.join(self.img_path, img_id)
            if not os.path.isfile(filename):
                raise ValueError(f"Image path doesn't exist: {filename}")

            generated_text_prefix = self.single_predict(filename)
            res_captions.append({"image_id": img_id, "caption": generated_text_prefix})
            test_progress.update()

        return res_captions


def save_predicted(preds: list, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(preds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ViT_MLP_both")
    parser.add_argument(
        "--clip_model_type",
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
    )
    parser.add_argument("--image_path", default="./deepfashion-mm/images")
    parser.add_argument(
        "--data_path", default="./deepfashion-mm/data_splits_ViT-B_32.json"
    )
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument(
        "--use_beam_search", dest="use_beam_search", action="store_true"
    )
    parser.add_argument("--only_prefix", dest="only_prefix", action="store_true")
    parser.add_argument(
        "--model_path",
        default="./checkpoints/clipcap_20220611_213502/deepfashion_clipcap-009.pt",
    )
    args = parser.parse_args()

    # MLP
    weights_path = args.model_path

    if args.only_prefix:
        model = ClipCaptionPrefix(
            args.prefix_length,
            clip_length=40,
            prefix_size=640,
            num_layers=8,
            mapping_type="transformer",
        )
    else:
        model = ClipCaptionModel(args.prefix_length)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model = model.eval()
    model = model.to(DEVICE)

    predictor = Predictor(model, args)
    res_captions = predictor.predict()

    output_dir = os.path.join(os.getcwd(), "/output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f"{output_dir}/pred_captions_{args.model_name}.json"
    save_predicted(res_captions, output_file)
    print(f"{len(res_captions)} predicted captions saved in {output_file}.")
