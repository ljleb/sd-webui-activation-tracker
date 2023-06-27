import dataclasses
import gradio as gr
import torch
from modules import scripts, script_callbacks, shared
from typing import Optional


class ActivationScript(scripts.Script):
    def title(self):
        return 'Activation Tracker'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Activation Tracker', open=False):
            with gr.Row():
                gr_start_steps_ratio = gr.Slider(
                    label='Accumulation Start Step Ratio',
                    value=0,
                    minimum=0,
                    maximum=1,
                )
                gr_end_steps_ratio = gr.Slider(
                    label='Accumulation End Step Ratio',
                    value=1,
                    minimum=0,
                    maximum=1,
                )

        return [gr_start_steps_ratio, gr_end_steps_ratio]

    def process(self, p, local_start_steps_ratio, local_end_steps_ratio):
        global active, start_steps_ratio, end_steps_ratio
        active = True
        start_steps_ratio = local_start_steps_ratio
        end_steps_ratio = local_end_steps_ratio

        for key, parameter in p.sd_model.named_parameters():
            if 'model.diffusion_model' in key:
                if unwrap_key(key) not in state_dict:
                    state_dict[unwrap_key(key)] = torch.zeros_like(parameter.data, device=parameter.device)

                parameter.requires_grad_()
                parameter.retain_grad = True

        print('state dict len:', len(state_dict))

    def postprocess(self, p, processed, *args):
        global active
        active = False


def unwrap_key(key: str) -> str:
    return key.replace('.wrapped.', '.')


@dataclasses.dataclass
class ForwardHook:
    key: str
    no_grad: Optional[torch.set_grad_enabled] = None

    def pre_forward_callback(self, model, args, kwargs):
        self.no_grad = torch.set_grad_enabled(True)
        self.no_grad.__enter__()

        model.zero_grad()

    def post_forward_callback(self, model, args, kwargs, output):
        global accumulation_count

        if self.key == 'model.diffusion_model' and is_accumulation_step():
            loss = output.sum()
            loss.backward()

            for key, parameter in shared.sd_model.named_parameters():
                if not 'model.diffusion_model' in key:
                    continue

                unwrapped_key = unwrap_key(key)
                state_dict[unwrapped_key] = state_dict[unwrapped_key] * max(1, accumulation_count)
                state_dict[unwrapped_key] += parameter.grad / shared.state.sampling_steps
                state_dict[unwrapped_key] /= accumulation_count + 1

            accumulation_count += 1

        self.no_grad.__exit__(None, None, None)
        self.no_grad = None


def is_accumulation_step():
    current_steps_ratio = (shared.state.sampling_step + 1) / shared.state.sampling_steps
    return start_steps_ratio <= current_steps_ratio < end_steps_ratio


def on_model_loaded(sd_model):
    for hooks in (pre_hooks, post_hooks):
        for hook in hooks.values():
            hook.remove()

        hooks.clear()

    for name, module in sd_model.named_modules():
        hook = ForwardHook(name)
        pre_hooks[name] = module.register_forward_pre_hook(hook.pre_forward_callback, with_kwargs=True)
        post_hooks[name] = module.register_forward_hook(hook.post_forward_callback, with_kwargs=True)


script_callbacks.on_model_loaded(on_model_loaded)


active = False
state_dict = {}
accumulation_count = 0
start_steps_ratio = 0
end_steps_ratio = 1
pre_hooks = {}
post_hooks = {}


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
