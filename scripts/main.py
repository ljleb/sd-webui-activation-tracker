import dataclasses
import torch
from modules import scripts, script_callbacks, shared

active = False


class TorchNoGradHijack:
    def __call__(self, f):
        # annotation hijack
        if not active:
            return original_torch_no_grad()(f)
        return f

    def __enter__(self):
        if not active:
            self.no_grad = original_torch_no_grad()
            return self.no_grad.__enter__()
        torch.set_grad_enabled(True)
        return self

    def __exit__(self, *args, **kwargs):
        if not active:
            return self.no_grad.__exit__(*args, **kwargs)


original_torch_no_grad = torch.no_grad
torch.no_grad = TorchNoGradHijack


class ActivationScript(scripts.Script):
    def __init__(self):
        self.state_dict = {}
        self.accumulation_count = 0

    def title(self):
        return "Activation"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        global active
        active = True
        for key, parameter in p.sd_model.named_parameters():
            if 'model.diffusion_model' in key:
                self.state_dict[unwrap_key(key)] = torch.zeros_like(parameter.data, device=parameter.device)
                parameter.requires_grad_()
                parameter.retain_grad = True

        print('state dict len:', len(self.state_dict))

    def postprocess_batch(self, p, *args, **kwargs):
        for key, parameter in p.sd_model.named_parameters():
            if parameter.grad is not None:
                unwrapped_key = unwrap_key(key)
                self.state_dict[unwrapped_key] = self.state_dict[unwrapped_key] * max(1, self.accumulation_count) / (self.accumulation_count + 1)
                self.state_dict[unwrapped_key] += parameter.grad / p.steps / (self.accumulation_count + 1)

        self.accumulation_count += 1

    def postprocess(self, p, processed, *args):
        global active
        active = False


def unwrap_key(key: str) -> str:
    return key.replace('.wrapped.', '.')


pre_hooks = {}
post_hooks = {}


def on_model_loaded(sd_model):
    global pre_hooks, post_hooks
    for hooks in [pre_hooks, post_hooks]:
        for hook in hooks.values():
            hook.remove()

        hooks.clear()

    for name, module in sd_model.named_modules():
        hook = ForwardHook(name)
        pre_hooks[name] = module.register_forward_pre_hook(hook.pre_forward_callback, with_kwargs=True)
        post_hooks[name] = module.register_forward_hook(hook.post_forward_callback, with_kwargs=True)

    root_hook = ForwardHook('root')
    pre_hooks['root'] = sd_model.register_forward_pre_hook(root_hook.pre_forward_callback, with_kwargs=True)
    post_hooks['root'] = sd_model.register_forward_hook(root_hook.post_forward_callback, with_kwargs=True)


script_callbacks.on_model_loaded(on_model_loaded)


@dataclasses.dataclass
class ForwardHook:
    key: str
    ctx_mgr: ... = None

    def pre_forward_callback(self, model, args, kwargs):
        self.ctx_mgr = torch.set_grad_enabled(True)
        self.ctx_mgr.__enter__()

    def post_forward_callback(self, model, args, kwargs, output):
        if self.key == 'model.diffusion_model':
            the_sum = output.sum()
            the_sum.backward()

        self.ctx_mgr.__exit__(None, None, None)
        self.ctx_mgr = None
