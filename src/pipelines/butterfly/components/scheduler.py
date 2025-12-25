from diffusers import DDPMScheduler

def get_scheduler(steps = 1000, beta = "squaredcos_cap_v2") ->DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=steps, beta_schedule=beta)