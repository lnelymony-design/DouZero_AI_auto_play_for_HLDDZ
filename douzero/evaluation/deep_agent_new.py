import numpy as np
import torch

from douzero.env.env_new import get_obs


def _resolve_device(device=None):
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for DeepAgent but torch.cuda.is_available() is False")
    return resolved


def _load_model(position, model_path, model_type, device=None):
    from douzero.dmc.models_new import (
        model_dict,
        model_dict_resnet,
        model_dict_general,
    )

    if model_type == "general":
        model = model_dict_general[position]()
    elif model_type == "resnet":
        model = model_dict_resnet[position]()
    else:
        model = model_dict[position]()

    resolved_device = _resolve_device(device)
    model_state_dict = model.state_dict()
    pretrained = torch.load(model_path, map_location=resolved_device)

    pretrained = {
        key: value
        for key, value in pretrained.items()
        if key in model_state_dict
    }
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


class DeepAgent:
    def __init__(self, position, model_path, device=None):
        self.model_type = "old"

        if "general" in model_path:
            self.model_type = "general"
        elif "resnet" in model_path:
            self.model_type = "resnet"

        self.model, self.device = _load_model(
            position,
            model_path,
            self.model_type,
            device=device,
        )

    def act(self, infoset):
        obs = get_obs(infoset, model_type=self.model_type)

        z_batch = torch.from_numpy(obs["z_batch"]).float().to(self.device)
        x_batch = torch.from_numpy(obs["x_batch"]).float().to(self.device)

        with torch.inference_mode():
            y_pred = self.model.forward(
                z_batch,
                x_batch,
                return_value=True,
            )["values"]

        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        best_action_confidence = y_pred[best_action_index]

        action_list = [
            (infoset.legal_actions[i], y_pred[i])
            for i in range(len(infoset.legal_actions))
        ]
        return best_action, best_action_confidence, action_list
