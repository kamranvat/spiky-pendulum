import torch
import snntorch
from rstdp import RSTDP


class Model(torch.nn.Module):
    def __init__(
        self, input_size: int, output_size: int, time_steps_per_action: int = 50
    ):
        super().__init__()

        # Try cuda, then mps, then cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.spike_time = time_steps_per_action
        self.optim = None

         # bias = False is recommended? for RSTDP optimiser
        self.con1 = torch.nn.Linear(input_size, 100, bias=False) 
        self.lif1 = snntorch.Leaky(0.9, 0.5, surrogate_disable=True)
        self.con2 = torch.nn.Linear(100, 100, bias=False)
        self.lif2 = snntorch.Leaky(0.9, 0.5, surrogate_disable=True)
        self.con3 = torch.nn.Linear(100, output_size, bias = False)
        self.lif3 = snntorch.Leaky(0.9, 0.5, surrogate_disable = True)

        print(
            f"Model initialized with input size {input_size} using device: {self.device}"
        )

    def forward(
        self, x: torch.Tensor, use_traces: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_traces and self.optim is None:
            raise RuntimeError(f"You have to set the optimiser first to return traces.")

        mem_rec = []
        spk_rec = []
        for step in range(self.spike_time):
            cur = self.con1(x[step])
            spk1, _ = self.lif1(cur)
            cur2 = self.con2(spk1)
            spk2, _ = self.lif2(cur2)
            cur3 = self.con3(spk2)
            spk3, mem3 = self.lif3(cur3)

            mem_rec.append(mem3)
            spk_rec.append(spk3)

            if use_traces:
                self.optim.update_e_trace(
                    pre_firing=[[x[step], spk1, spk2]], post_firing=[[spk1, spk2, spk3]]
                )

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def set_optim(self, lr: float = 0.01, **kwargs):
        self.optim = RSTDP(
            self.parameters(), time_steps=self.spike_time, lr=lr, **kwargs
        )
