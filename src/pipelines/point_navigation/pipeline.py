from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers.utils import BaseOutput

from src.pipelines.point_navigation.components.model import TrajectoryDiffusionModel


class TrajectoryPipelineOutput(BaseOutput):
    trajectories: torch.Tensor


class PointNavigationPipeline(DiffusionPipeline):
    model: TrajectoryDiffusionModel
    scheduler: DDPMScheduler

    def __init__(self, model: TrajectoryDiffusionModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            start: Union[List[float], Tuple[float, float]],
            target: Union[List[float], Tuple[float, float]],
            batch_size: int = 1,
            num_inference_steps: int = 1000,
            generator: Optional[torch.Generator] = None,
    ) -> TrajectoryPipelineOutput:
        device = self.device

        observation = torch.tensor(
            [[start[0], start[1], target[0], target[1]]] * batch_size,
            device=device,
            dtype=torch.float32
        )

        trajectory = torch.randn(
            (batch_size, 32, 2),
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            noise_pred = self.model(
                trajectory,
                torch.tensor([t] * batch_size, device=device),
                observation,
            )
            trajectory = self.scheduler.step(noise_pred, t, trajectory).prev_sample

        return TrajectoryPipelineOutput(trajectories=trajectory)
