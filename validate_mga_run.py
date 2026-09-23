# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Validate completed execution and stage checkpoints for MNIST, CIFAR-10 or Purchase.
This is not a significance test or a guarantee of historical unlearning.
"""
from pathlib import Path
import argparse
from collections import Counter
import json

import torch
import yaml

from mga.evaluation import checkpoint_name, dataset_name, load_test_dataset, evaluate_checkpoint


def validate(run, device):
    cfg = yaml.safe_load((run / "config.yml").read_text())
    name = dataset_name(cfg)
    model_name = checkpoint_name(cfg)
    status = json.loads((run / "results/mga_status.json").read_text())
    if status["status"] != "completed": raise ValueError("MGA run did not complete.")
    targets = set(cfg["clients"]["clients_requesting_deletion"])
    executions = [json.loads(p.read_text()) for p in (run / "results/mga_clients").glob("*.execution.json")]
    reverse = [e for e in executions if e["phase"] == "unlearn"]
    if Counter(e["client_id"] for e in reverse) != Counter({i:1 for i in targets}):
        raise ValueError("Each target must complete exactly one reverse dispatch.")
    if any(e["reverse"]["gradient_examples"] <= 0 for e in reverse):
        raise ValueError("An unlearning dispatch did not compute a gradient.")
    if any(e["client_id"] in targets for e in executions if e["phase"] == "recover"):
        raise ValueError("Target client reentered recovery.")
    if any(e["poison_reads"] != 0 for e in executions if e["client_id"] not in targets):
        raise ValueError("Retained-client data were poisoned.")
    train_poison_reads = sum(e["poison_reads"] for e in executions if e["phase"] == "train")
    if cfg["clients"].get("backdoor_enabled") and cfg["clients"].get("backdoor_poison_rate",0)>0 and train_poison_reads<1:
        raise ValueError("Poisoning is configured but no poisoned training examples were read.")
    events = [json.loads(line) for line in (run / "results/mga_events.jsonl").read_text().splitlines()]
    completed = [i for i,e in enumerate(events) if e["event"] == "unlearning_complete"]
    returns = [i for i,e in enumerate(events) if e["event"] == "reverse_return"]
    recovery = [i for i,e in enumerate(events) if e["event"] == "recovery_selected"]
    if len(completed)!=1 or not returns or not recovery or max(returns)>=completed[0] or min(recovery)<=completed[0]:
        raise ValueError("Invalid return/completion/recovery ordering.")
    dataset = load_test_dataset(cfg)
    checkpoints={"initial":run/"initial_random_state.pth",
                 "before_request":run/"results/mga_before_request.pth",
                 "reference":run/"results/mga_reference.pth",
                 "after_unlearning":run/"results/mga_after_unlearning.pth",
                 "after_recovery":run/f"models/{model_name}.pth"}
    metrics={}
    for phase,path in checkpoints.items():
        metrics[phase] = evaluate_checkpoint(path, cfg, dataset, device)
    reference=torch.load(checkpoints["reference"],map_location="cpu",weights_only=True)
    after=torch.load(checkpoints["after_unlearning"],map_location="cpu",weights_only=True)
    reference_event=next(e for e in events if e["event"]=="reference_ready")
    expected={k:v.clone() for k,v in reference.items()}
    for i in sorted(targets):
        state=torch.load(run/f"models/{model_name}_{i}.pth",map_location="cpu",weights_only=True)
        weight=reference_event["return_weights"][str(i)]
        for key in expected:
            if expected[key].is_floating_point():expected[key].add_(state[key]-reference[key],alpha=weight)
    for key in expected:torch.testing.assert_close(expected[key],after[key],rtol=1e-6,atol=1e-7)
    result={"execution_valid":True,"dataset":name,"run":str(run),"targets":sorted(targets),
            "poisoned_training_reads":train_poison_reads,
            "reverse_dispatches":len(reverse),
            "reverse_steps":{e["client_id"]:e["reverse"]["accepted_steps"] for e in reverse},
            "recovery_dispatches":sum(e["phase"]=="recover" for e in executions),
            "reverse_outputs_included_in_aggregate":True,
            "metrics":metrics,"asr_protocol":"full non-target test set; raw-space inversion then training normalization",
            "warning":"Execution validation, not proof of retraining equivalence or reproduction of old benchmark values."}
    (run/"validated_result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args();torch.set_num_threads(2)
    print(json.dumps(validate(Path(args.run).resolve(),args.device),indent=2))
