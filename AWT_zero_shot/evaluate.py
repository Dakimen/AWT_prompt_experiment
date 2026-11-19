import argparse
import time
import os
import json
import csv  # Added for experiment
import hashlib  # Added for experiment

import torch
import torch.nn.functional as F

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES

from clip import clip
def load_clip_to_cpu(arch):
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

@torch.no_grad()
def get_entropy_weight(output, img_t=0.5, text_t=0.5):
    with torch.cuda.amp.autocast():
        # get weights for images
        image_entropy = calculate_batch_entropy(output.mean(1))
        image_weights = F.softmax(-image_entropy/img_t, dim=-1)

        # get weights for descriptors
        _, n_des, n_cls = output.shape
        anchor = output[0].mean(0)[None, None, :].repeat(n_des, n_cls, 1)
        output_des = output[0].unsqueeze(-1)
        scatter_indices = torch.arange(n_cls)[None, :, None].repeat(n_des, 1, 1).cuda()
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des) # n_des, n_cls, n_cls
        text_entropy = calculate_batch_entropy(anchor)
        text_weights = F.softmax(-text_entropy.t()/text_t, dim=-1) # n_cls, n_des

    return image_weights, text_weights

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def optimal_transport(logits, logit_scale, image_weights, text_weights):
    eps = 0.1
    sim = logits / logit_scale.exp()
    sim = sim.permute(2, 0, 1) # n_cls x M x N

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.permute(1, 2, 0)
    assert not torch.isnan(T).any()

    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0)


@torch.no_grad()
def AWT_evaluation(clip_model, args):
    dataset_name = args.test_set
    print("Evaluating: {}".format(dataset_name))
    classnames = get_classnames(dataset_name)

    # get LLM descriptors
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    llm_descriptions = json.load(open(description_file))

    # ============== prepare text features ==============
    text_features = []
    template = CUSTOM_TEMPLATES[dataset_name]
    for classname in classnames:
        prompts = []
        prompt = template.format(classname.replace("_", " "))
        prompts.append(prompt + '.')

        # get descriptions
        assert len(llm_descriptions[classname]) >= args.num_descriptor
        for i in range(args.num_descriptor):
            prompt_desc = prompt + '. ' + llm_descriptions[classname][i]
            prompts.append(prompt_desc)
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.cuda.amp.autocast():
            text_features.append(clip_model.encode_text(prompts)) # n_desc x d

    text_features = torch.cat(text_features).float() # (n_cls x n_desc) x d
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ==============  calculate logits for each image ==============
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    pre_features = torch.load(f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}/{dataset_name}.pth")

    results_dir = os.path.join(  # Start of added code block
            "experiment_results",
            args.condition
        )
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "predictions.csv")
    metrics_path = os.path.join(results_dir, "metrics.json")

    if os.path.exists(csv_path) or os.path.exists(metrics_path):
        raise FileExistsError(
            f"Results already exist for condition {args.condition}. "
                "Back them up or remove them before running again."
        )

    with open(description_file, "rb") as f:
        description_hash = hashlib.sha256(f.read()).hexdigest()

    per_image_results = []
    all_margins = []
    all_losses = []

    score_temperature = 0.1  # End of added code block

    print("number of test samples: {}".format(len(pre_features)))
    progress = ProgressMeter(len(pre_features), [batch_time, top1], prefix='Test: ')

    end = time.time()
    for i, (image_features, target) in enumerate(pre_features):
        n_views = image_features.size(0)
        n_prompt = args.num_descriptor + 1

        output = clip_model.logit_scale.exp() * image_features @ text_features.t()
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous() # n_view x n_prompt x c

        image_temperature = 0.5
        text_temperature = 0.5
        image_weights, text_weights = get_entropy_weight(output, img_t=image_temperature, text_t=text_temperature)
        output_ot = optimal_transport(output, clip_model.logit_scale, image_weights, text_weights)

        # measure accuracy
        acc1, = accuracy(output_ot, target, topk=(1,))
        top1.update(acc1[0], 1)

        scores = (  # Added code block
                output_ot.float() /
                clip_model.logit_scale.exp().float()
        ).squeeze(0)

        true_idx = int(target.item())
        pred_idx = int(scores.argmax().item())

        # Ground-truth score minus the strongest competing score.
        true_score = scores[true_idx]
        competitor_scores = scores.clone()
        competitor_scores[true_idx] = float("-inf")
        highest_competitor = competitor_scores.max()

        margin = (true_score - highest_competitor).item()

        # Softmax cross-entropy with a fixed evaluation temperature.
        loss = F.cross_entropy(
            (scores / score_temperature).unsqueeze(0),
            target.reshape(1).long()
        ).item()

        record = {
            "feature_index": i,
            "true_class": classnames[true_idx],
            "predicted_class": classnames[pred_idx],
            "correct": int(pred_idx == true_idx),
            "margin": margin,
            "cross_entropy": loss,
        }

        # Save the individual score for every class.
        for class_idx, classname in enumerate(classnames):
            record[f"score_{classname}"] = scores[class_idx].item()

        per_image_results.append(record)
        all_margins.append(margin)
        all_losses.append(loss)  # End of added code block

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    print(f'\n *  {dataset_name}')
    progress.display_summary()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:  # Start of added code block
        writer = csv.DictWriter(
            f,
            fieldnames=per_image_results[0].keys()
            )
        writer.writeheader()
        writer.writerows(per_image_results)
    
    num_samples = len(per_image_results)
    num_correct = sum(r["correct"] for r in per_image_results)
    
    per_class_accuracy = {}
    
    for classname in classnames:
        class_records = [
            r for r in per_image_results
            if r["true_class"] == classname
        ]
    
        correct = sum(r["correct"] for r in class_records)
    
        per_class_accuracy[classname] = {
            "correct": correct,
            "total": len(class_records),
            "accuracy_percent": 100.0 * correct / len(class_records)
        }
    
    metrics = {
        "condition": args.condition,
        "dataset": dataset_name,
        "architecture": args.arch,
        "seed": args.seed,
        "num_descriptors": args.num_descriptor,
        "description_sha256": description_hash,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy_percent": 100.0 * num_correct / num_samples,
        "mean_signed_margin": sum(all_margins) / num_samples,
        "mean_cross_entropy": sum(all_losses) / num_samples,
        "score_definition": "OT-weighted logits divided by CLIP logit scale",
        "cross_entropy_temperature": score_temperature,
        "per_class_accuracy": per_class_accuracy
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n===== Additional Experimental Metrics =====")
    print(f"Accuracy:          {metrics['accuracy_percent']:.2f}%")
    print(f"Mean signed margin: {metrics['mean_signed_margin']:.6f}")
    print(f"Mean cross-entropy: {metrics['mean_cross_entropy']:.6f}")
    print(f"Saved predictions:  {csv_path}")
    print(f"Saved metrics:      {metrics_path}")  # End of added code block


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model = load_clip_to_cpu(args.arch)
    clip_model = clip_model.cuda()
    clip_model.float()
    clip_model.eval()

    for _, param in clip_model.named_parameters():
        param.requires_grad_(False)
    
    # testing start
    AWT_evaluation(clip_model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT evaluation')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)