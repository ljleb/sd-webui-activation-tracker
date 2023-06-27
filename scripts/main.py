import dataclasses
import torch
from modules import scripts, script_callbacks


active = False


class TorchNoGradHijack:
    def __call__(self, f):
        if not active:
            return original_torch_no_grad()(f)
        return f

    def __enter__(self):
        if not active:
            return original_torch_no_grad().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


original_torch_no_grad = torch.no_grad
torch.no_grad = TorchNoGradHijack


class ActivationScript(scripts.Script):
    def title(self):
        return "Activation"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        global active
        active = True
        for parameter in p.sd_model.parameters():
            parameter.requires_grad = True

    def postprocess(self, p, processed, *args):
        global active
        active = False
        for key, parameter in p.sd_model.named_parameters():
            if parameter.grad is None:
                print(key)
            parameter.requires_grad = False


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


script_callbacks.on_model_loaded(on_model_loaded)


@dataclasses.dataclass
class ForwardHook:
    key: str
    ctx_mgr: ... = None

    def pre_forward_callback(self, model, args, kwargs):
        self.ctx_mgr = torch.set_grad_enabled(True)
        self.ctx_mgr.__enter__()

    def post_forward_callback(self, model, args, kwargs, output) -> None:
        if self.key == "first_stage_model.decoder":
            the_sum = output.sum()
            the_sum.retain_grad()
            the_sum.backward()
        self.ctx_mgr.__exit__(None, None, None)
        self.ctx_mgr = None
