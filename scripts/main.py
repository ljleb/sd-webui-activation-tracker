import dataclasses
import gradio as gr
import safetensors.torch
import torch
from lib_activation_tracker import global_state
from modules import scripts, script_callbacks, shared
from typing import Optional


class ActivationScript(scripts.Script):
    def title(self):
        return 'Activation Tracker'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Activation Tracker', open=False):
            enabled = gr.Checkbox(label='Enable')
            with gr.Row():
                start_steps_ratio = gr.Slider(
                    label='Accumulation start step ratio',
                    value=0,
                    minimum=0,
                    maximum=1,
                )
                end_steps_ratio = gr.Slider(
                    label='Accumulation end step ratio',
                    value=1,
                    minimum=0,
                    maximum=1,
                )
            with gr.Row():
                save_activations = gr.Button(
                    value='Save activations',
                )
                reset_activations = gr.Button(
                    value='Clear activations'
                )

            save_activations.click(
                fn=self.save_activations
            )
            reset_activations.click(
                fn=self.reset_activations
            )

        return [enabled, start_steps_ratio, end_steps_ratio]

    def save_activations(self):
        safetensors.torch.save_file(global_state.activations, r'E:\sd\models\activation\unet.safetensors')

    def reset_activations(self):
        global_state.activations.clear()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def process(self, p, enabled, start_steps_ratio, end_steps_ratio):
        global_state.enabled = enabled
        if not enabled:
            return

        global_state.start_steps_ratio = start_steps_ratio
        global_state.end_steps_ratio = end_steps_ratio

        for key, parameter in p.sd_model.named_parameters():
            if 'model.diffusion_model' in key:
                if unwrap_key(key) not in global_state.activations:
                    global_state.activations[unwrap_key(key)] = torch.zeros_like(parameter.data, device=parameter.device)

                parameter.requires_grad_()
                parameter.retain_grad = True

        print('state dict len:', len(global_state.activations))

    def postprocess(self, p, processed, *args):
        global_state.enabled = False


def unwrap_key(key: str) -> str:
    return key.replace('.wrapped.', '.')


@dataclasses.dataclass
class BackwardHook:
    key: str
    no_grad: Optional[torch.set_grad_enabled] = None

    def pre_forward_callback(self, model, args, kwargs):
        if not global_state.enabled:
            return

        self.no_grad = torch.set_grad_enabled(True)
        self.no_grad.__enter__()

        model.zero_grad()

    def post_forward_callback(self, model, args, kwargs, output):
        if not global_state.enabled:
            return

        if self.key == 'model.diffusion_model' and is_accumulation_step():
            loss = output.sum()
            loss.backward()

            for key, parameter in shared.sd_model.named_parameters():
                if not 'model.diffusion_model' in key:
                    continue

                unwrapped_key = unwrap_key(key)
                global_state.activations[unwrapped_key] = global_state.activations[unwrapped_key] * max(1, global_state.accumulation_count)
                global_state.activations[unwrapped_key] += parameter.grad / shared.state.sampling_steps
                global_state.activations[unwrapped_key] /= global_state.accumulation_count + 1

            model.zero_grad()
            global_state.accumulation_count += 1

        self.no_grad.__exit__(None, None, None)
        self.no_grad = None


def is_accumulation_step():
    current_steps_ratio = (shared.state.sampling_step + 1) / shared.state.sampling_steps
    return global_state.start_steps_ratio <= current_steps_ratio < global_state.end_steps_ratio


def on_model_loaded(sd_model):
    for hooks in (global_state.pre_hooks, global_state.post_hooks):
        for hook in hooks.values():
            hook.remove()

        hooks.clear()

    for name, module in sd_model.named_modules():
        hook = BackwardHook(name)
        global_state.pre_hooks[name] = module.register_forward_pre_hook(hook.pre_forward_callback, with_kwargs=True)
        global_state.post_hooks[name] = module.register_forward_hook(hook.post_forward_callback, with_kwargs=True)


script_callbacks.on_model_loaded(on_model_loaded)


class TorchNoGradHijack:
    def __call__(self, f):
        # annotation hijack
        if not global_state.enabled:
            return original_torch_no_grad()(f)
        return f

    def __enter__(self):
        if not global_state.enabled:
            self.no_grad = original_torch_no_grad()
            return self.no_grad.__enter__()
        torch.set_grad_enabled(True)
        return self

    def __exit__(self, *args, **kwargs):
        if not global_state.enabled:
            return self.no_grad.__exit__(*args, **kwargs)


original_torch_no_grad = torch.no_grad
torch.no_grad = TorchNoGradHijack
