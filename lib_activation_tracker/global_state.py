enabled = False
activations = {}
accumulation_steps = {}
start_steps_ratio = 0
end_steps_ratio = 1
pre_hooks = {}
post_hooks = {}

# the webui does a bad job of setting these values, we maintain our own
generation_current_step = 0
generation_total_steps = 0
