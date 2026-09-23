# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Explicit normalized reverse optimization for the corrected MGA dispatch.

One iteration accumulates the mean loss gradient over the supplied client data.
BatchNorm buffers and dropout are held fixed (eval mode, gradients still enabled).
This path uses scalar normalized ascent, no momentum/weight decay, and rejects
an infeasible proposal. The separate historical trainer path is not rewritten.
"""
from __future__ import annotations

import math
import torch


def reverse_optimize(model, reference, loader, criterion, iterations, lr, radius, device):
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("MGA requires a positive integer iteration budget.")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("MGA learning rate and radius must be finite and positive.")
    model.load_state_dict(reference, strict=True)
    model.to(device)
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    centers = [p.detach().clone() for p in params]
    result = {"assigned_steps": int(iterations), "accepted_steps": 0,
              "attempted_steps": 0, "stop_reason": "budget", "radius": float(radius),
              "step_size": float(lr), "normalized_step_length": lr * math.sqrt(iterations),
              "trace": [], "gradient_examples": 0}

    def mean_loss(backward=False):
        total = 0
        loss_sum = 0.0
        for examples, labels in loader:
            examples, labels = examples.to(device), labels.to(device)
            n = int(labels.shape[0])
            loss = criterion(model(examples), labels)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("MGA requires a finite scalar mean loss.")
            if backward:
                (loss * n).backward()
            total += n
            loss_sum += float(loss.detach()) * n
        if not total:
            raise ValueError("No examples were supplied to reverse optimization.")
        return loss_sum / total, total

    try:
        for step in range(int(iterations)):
            model.zero_grad(set_to_none=True)
            loss, n = mean_loss(backward=True)
            if step == 0:
                result["loss_before"] = loss
            result["gradient_examples"] += n
            for p in params:
                if p.grad is not None:
                    p.grad.div_(n)
            squared = sum(float(p.grad.detach().double().square().sum())
                          for p in params if p.grad is not None)
            norm = math.sqrt(squared)
            if not math.isfinite(norm):
                raise ValueError("Non-finite MGA gradient.")
            if norm == 0:
                result["stop_reason"] = "zero_gradient"
                break
            previous = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=lr * math.sqrt(iterations) / norm)
                distance = math.sqrt(sum(float((p - c).double().square().sum())
                                         for p, c in zip(params, centers)))
            result["attempted_steps"] += 1
            accepted = math.isfinite(distance) and distance <= radius
            result["trace"].append({"step": step + 1, "mean_loss": loss,
                                    "gradient_norm": norm,
                                    "proposal_distance": distance, "accepted": accepted})
            if not accepted:
                with torch.no_grad():
                    for p, old in zip(params, previous):
                        p.copy_(old)
                result["stop_reason"] = "outside_radius"
                break
            result["accepted_steps"] += 1
        with torch.no_grad():
            result["loss_after"], _ = mean_loss()
            result["returned_distance"] = math.sqrt(sum(
                float((p - c).double().square().sum()) for p, c in zip(params, centers)))
        if result["returned_distance"] > radius + 1e-7:
            raise RuntimeError("Returned MGA parameters are outside the feasible region.")
        return result
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
