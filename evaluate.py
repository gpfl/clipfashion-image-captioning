import argparse
import json

from nlgeval import compute_metrics


def read_json(file_path: str):
    with open(file_path, "r") as f:
        out = json.load(f)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_path", default="./deepfashion-mm/captions.json")
    parser.add_argument("--pred_path", type=str)
    args = parser.parse_args()

    captions = read_json(args.cap_path)
    preds = read_json(args.pred_path)

    pred_caption = [pred["caption"] + "\n" for pred in preds]
    ref_caption = [captions[pred["image_id"]] for pred in preds]

    with open("output/ref_captions.txt", "w") as f:
        for pred in preds:
            f.write(captions[pred["image_id"]])
            f.write("\n")

    with open("output/pred_captions_ViT_MLP_both.txt", "w") as f:
        for pred in preds:
            f.write(pred["caption"])
            f.write("\n")

    metrics_dict = compute_metrics(
        hypothesis="".join(pred_caption),
        references=ref_caption,
    )
    print(metrics_dict[:5])
