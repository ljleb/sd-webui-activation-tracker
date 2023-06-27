import dataclasses

from modules import script_callbacks, scripts
import torch


class ActivationScript(scripts.Script):
    def title(self):
        return "Activation"

    def process(self, p, *args):
        for parameter in p.sd_model.parameters():
            parameter.requires_grad = True

    def postprocess(self, p, processed, *args):
        for parameter in p.sd_model.parameters():
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

    def pre_forward_callback(self, model, args, kwargs):
        self.ctx_mgr = torch.set_grad_enabled(True)
        self.ctx_mgr.__enter__()

    def post_forward_callback(self, model, args, kwargs, output) -> None:
        if self.key == "first_stage_model.decoder":
            output.sum().backward()
        self.ctx_mgr.__exit__(None, None, None)
