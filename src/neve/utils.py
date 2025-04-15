import copy
from typing import List, Dict

from torch import nn

from src.neve.hooks import Hook


def mse(a, b):
    return ((a - b) ** 2).mean()


def calculate_metrics(old_metrics: List, new_metrics: List) -> Dict:
    result = {
        "model_metric": float("Inf"),
        "model_metric_avg": float("Inf"),
        "model_metric_mse": None,
        "layers_metric_mse": None
    }

    # Update velocities if vector 'new_velocities' is not null
    if new_metrics is not None:
        result["model_metric"] = sum([sum(abs(vels)) for vels in new_metrics])
        result["model_metric_avg"] = result["model_metric"] / sum([len(vals) for vals in new_metrics])

    if not (old_metrics is None or new_metrics is None
            or len(old_metrics) != len(new_metrics)
            or len(old_metrics) == 0):
        mses = [mse(val1, val2) for val1, val2 in zip(old_metrics, new_metrics)]
        result["model_metric_mse"] = sum(mses) / len(mses)
        result["layers_metric_mse"] = mses
    return result


def activate_hooks(hooks: Dict[str, Hook], active: bool):
    for h in hooks:
        hooks[h].activate(active)


def attach_hooks(args, model) -> Dict[str, Hook]:
    hooks = {}
    # Hook all layers
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            hooks[n] = Hook(n, m, args.velocity_mu, args.neve_use_gpu)
    print(f"Initialized {len(hooks)} hooks.")
    return hooks


def evaluate_velocity(hooks, logs, old_metrics):
    neve_new_metrics = []
    for k in hooks:
        # Get layers velocity from the hooks
        velocity = copy.deepcopy(hooks[k].get_velocity())
        # Log velocity histogram
        logs["velocity"][f"{k}"] = velocity
        # Save this epoch velocity for the next iteration
        neve_new_metrics.append(velocity)

    # Evaluate the velocity mse
    metrics_log_data = {
        "velocity": calculate_metrics(old_metrics, neve_new_metrics)
    }
    print("VELOCITY DATA:\n", metrics_log_data)

    # Log velocity
    metric_data = metrics_log_data["velocity"]
    logs["neve"]["model_value"] = metric_data["model_metric"]
    logs["neve"]["model_avg_value"] = metric_data["model_metric_avg"]
    # Log model overall value
    if metric_data["model_metric_mse"] is not None:
        logs["neve"]["model_mse_value"] = metric_data["model_metric_mse"]
    # Log for each layer the overall value
    if metric_data["layers_metric_mse"] is not None:
        logs["neve"]["layers_mse_value"] = {}
        for count, layer_value in enumerate(metric_data["layers_metric_mse"]):
            logs["neve"]["layers_mse_value"][str(count)] = layer_value
    return metrics_log_data, neve_new_metrics


def update_schedulers(schedulers, scheduler_metric, vloss, velocity_data, consider_baseline_scheduler=False):
    if scheduler_metric == "vloss":
        for scheduler in schedulers:
            scheduler.step(vloss)
    elif scheduler_metric == "neve":
        # If we use one scheduler x layer
        if velocity_data["model_metric_mse"]:
            schedulers[0].step(velocity_data["model_metric_mse"])
    # baseline scheduler
    else:
        if consider_baseline_scheduler:
            for scheduler in schedulers:
                if scheduler is not None:
                    scheduler.step()
