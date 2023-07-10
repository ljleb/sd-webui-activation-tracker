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
                clear_activations = gr.Button(
                    value='Clear activations'
                )

            save_activations.click(
                fn=self.save_activations
            )
            clear_activations.click(
                fn=self.clear_activations
            )

        return [enabled, start_steps_ratio, end_steps_ratio]

    def save_activations(self):
        print(fr'Saving activations to E:\sd\models\activation\unet.safetensors')
        safetensors.torch.save_file(global_state.activations, r'E:\sd\models\activation\unet.safetensors')

    def clear_activations(self):
        global_state.activations.clear()
        global_state.accumulation_steps.clear()
        with torch.no_grad():
            torch.cuda.empty_cache()
        print(fr'Cleared activations')

    def process(self, p, enabled, start_steps_ratio, end_steps_ratio):
        global_state.enabled = enabled
        if not enabled:
            return

        global_state.start_steps_ratio = start_steps_ratio
        global_state.end_steps_ratio = end_steps_ratio

        for key, parameter in p.sd_model.named_parameters():
            if is_backward_key(key):
                if unwrap_key(key) not in global_state.activations:
                    global_state.activations[unwrap_key(key)] = torch.zeros_like(parameter.data, device=parameter.device)

                parameter.requires_grad_()
                parameter.retain_grad = True

        print('state dict len:', len(global_state.activations))

    def process_batch(self, p, *args, **kwargs):
        global_state.generation_total_steps = p.steps
        global_state.generation_current_step = 0

    def postprocess(self, p, processed, *args):
        global_state.enabled = False
        for key, parameter in p.sd_model.named_parameters():
            if is_backward_key(key):
                # TODO: restore old `requires_grad` and `retain_grad`
                parameter.requires_grad = False
                parameter.retain_grad = False

        with torch.no_grad():
            torch.cuda.empty_cache()


def unwrap_key(key: str) -> str:
    return key.replace('.wrapped.', '.')


def is_backward_key(key: str) -> bool:
    return key.startswith(('model.diffusion_model', 'cond_stage_model')) and 'time_embed' not in key


def is_root_key(key: str) -> bool:
    return key in ('model.diffusion_model', 'cond_stage_model')


@dataclasses.dataclass
class BackwardHook:
    key: str
    no_grad: Optional[torch.set_grad_enabled] = None

    def pre_forward_callback(self, model, args, kwargs):
        if not global_state.enabled:
            return

        self.no_grad = torch.set_grad_enabled(True)
        self.no_grad.__enter__()

    def post_forward_callback(self, model, args, kwargs, output):
        if not global_state.enabled:
            return

        if is_root_key(self.key) and is_enabled_accumulation_step():
            loss = -output.mean()
            loss.backward()

            for key, parameter in shared.sd_model.named_parameters():
                if not is_backward_key(key) or not key.startswith(self.key):
                    continue

                unwrapped_key = unwrap_key(key)
                accumulation_step = global_state.accumulation_steps.get(unwrapped_key, 0)
                global_state.accumulation_steps[unwrapped_key] = accumulation_step + 1
                # dividing by `shared.state.sampling_steps` is wrong, needs to be the number of steps where `is_active_accumulation_step()` returns true
                # global_state.activations[unwrapped_key] = global_state.activations[unwrapped_key] * max(1, accumulation_step)
                # global_state.activations[unwrapped_key] += parameter.grad
                # global_state.activations[unwrapped_key] /= accumulation_step + 1

                # grad = torch.softmax(torch.flatten(parameter.grad), dim=0).reshape_as(parameter.grad)
                # global_state.activations[unwrapped_key] **= accumulation_step / (accumulation_step + 1)
                # global_state.activations[unwrapped_key] *= grad ** (1 / (accumulation_step + 1))
                # if not torch.nonzero(global_state.activations[unwrapped_key]).any():
                #     print('zero key', key)
                # if torch.isnan(global_state.activations[unwrapped_key]).any():
                #     print('nan key', key)

                grad = parameter.grad
                global_state.activations[unwrapped_key] **= accumulation_step / (accumulation_step + 1)
                global_state.activations[unwrapped_key] *= grad ** (1 / (accumulation_step + 1))
                if not torch.nonzero(global_state.activations[unwrapped_key]).any():
                    print('zero key', key)
                if torch.isnan(global_state.activations[unwrapped_key]).any():
                    print('nan key', key)

            model.zero_grad()

        self.no_grad.__exit__(None, None, None)
        self.no_grad = None


def is_enabled_accumulation_step():
    current_steps_ratio = global_state.generation_current_step / global_state.generation_total_steps
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


def on_cfg_denoised(*args, **kwargs):
    global_state.generation_current_step += 1


script_callbacks.on_cfg_denoised(on_cfg_denoised)


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
