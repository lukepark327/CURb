import torch


def canonical_lora_key(module_prefix: str) -> str:
    marker = "layers."
    idx = module_prefix.find(marker)
    if idx >= 0:
        return module_prefix[idx:]
    return module_prefix


def collect_lora_factors(model):
    lora_a = {}
    lora_b = {}
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if ".lora_A." in name:
            prefix = name.split(".lora_A.", 1)[0]
            lora_a[canonical_lora_key(prefix)] = param
        elif ".lora_B." in name:
            prefix = name.split(".lora_B.", 1)[0]
            lora_b[canonical_lora_key(prefix)] = param
    return lora_a, lora_b


def build_olora_prev_device_map(method_ctx, device, dtype):
    prev_map = method_ctx.get("olora_prev_A", {}) or {}
    device_map = {}
    for key, tensor in prev_map.items():
        if tensor is None or tensor.numel() == 0:
            continue
        device_map[key] = tensor.to(
            device=device,
            dtype=dtype,
            non_blocking=(device.type == "cuda"),
        )
    return device_map


def append_olora_subspace(method_ctx, lora_a):
    prev_map = method_ctx.setdefault("olora_prev_A", {})
    for key, tensor in lora_a.items():
        cur = tensor.detach().to(device="cpu", dtype=torch.float32)
        if key in prev_map and prev_map[key] is not None and prev_map[key].numel() > 0:
            prev_map[key] = torch.cat([prev_map[key], cur], dim=0)
        else:
            prev_map[key] = cur
    return prev_map


def compute_olora_losses(loss_like, olora_a, olora_b, olora_prev_a):
    orth_loss = loss_like.new_zeros(())
    l2_loss = loss_like.new_zeros(())

    for key, current_a in olora_a.items():
        prev_a = olora_prev_a.get(key)
        if prev_a is None or prev_a.numel() == 0:
            continue
        if prev_a.shape[1] != current_a.shape[1]:
            continue
        orth_loss = orth_loss + torch.abs(prev_a.to(dtype=current_a.dtype) @ current_a.t()).sum()

    for current_a in olora_a.values():
        l2_loss = l2_loss + torch.norm(current_a, p=2)
    for current_b in olora_b.values():
        l2_loss = l2_loss + torch.norm(current_b, p=2)

    return orth_loss, l2_loss

