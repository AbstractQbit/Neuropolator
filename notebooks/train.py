if __name__ == "__main__":
    import itertools
    import os
    import pickle
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

    import intel_extension_for_pytorch as ipex
    import schedulefree
    from tqdm import tqdm, trange

    from replay_loading import enum_replay_folder, files_to_strokes, sample_stroke
    from stroke_dataset import StrokeDataset, collate_simple_stack, default_transforms

    device = torch.device("xpu")

    batch_size = 256
    seq_len = 4096

    # replay_fns = list(enum_replay_folder("H:/osu!/Data/r/"))
    # all_strokes = list(files_to_strokes(tqdm(replay_fns), min_length=50))

    all_strokes = pickle.load(open("all_strokes.pkl", "rb"))

    ds_full = StrokeDataset(all_strokes, **default_transforms(seq_len=seq_len))
    ds_full_loader = DataLoader(
        ds_full, batch_size=batch_size, sampler=ds_full.wrand_sampler, collate_fn=collate_simple_stack, num_workers=12
    )

    class TestNet(torch.jit.ScriptModule):
        def __init__(self, kernels=[5] * 12, channels=[8] * 4 + [4] * 4 + [2] * 4, dilations=None, out_steps=5):
            super().__init__()

            self.kernels = torch.tensor(kernels + [1])
            self.out_steps = out_steps
            self.channels = torch.tensor(channels + [2 * out_steps])
            self.in_channels = torch.tensor([2] + channels).cumsum(dim=0)
            self.total_channels = self.in_channels[-1].item() + 2
            self.dilations = torch.tensor(dilations + [1]) if dilations is not None else None
            self.pads = (self.kernels - 1) * (self.dilations if self.dilations is not None else 1)
            self.pad_total = self.pads.sum().item()
            self.pad_max = self.pads.max().item()
            self.ar_len = 5

            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.in_channels[i].item(),
                        out_channels=self.channels[i].item() * 2,
                        kernel_size=self.kernels[i].item(),
                        dilation=self.dilations[i].item() if self.dilations is not None else 1,
                    )
                    for i in range(len(self.kernels))
                ]
            )
            self.n_layers = len(self.convs)

            print(f"Kernels: {self.kernels}")
            print(f"Input channels: {self.in_channels}")
            print(f"Channels: {self.channels}")
            print(f"Dilations: {self.dilations}")
            print(f"Pads: {self.pads}")
            print(f"Total padding: {self.pad_total}")

        @torch.jit.script_method
        def forward(self, x):
            # input is (batch, channels, seq_len)
            B, _, L = x.shape
            curr_window = torch.tensor(x.shape[-1])
            acts = [x]
            for i, conv in enumerate(self.convs):
                x = torch.cat([act[..., -curr_window:] for act in acts], dim=1)
                x = conv(x)
                x = F.glu(x, dim=1)
                curr_window -= self.pads[i]
                acts.append(x)
            return x.view(B, 2, self.out_steps, L - self.pad_total)

    dilated_params = {
        "kernels": [7, 7, 7, 5, 5, 5, 3, 3, 3],
        "channels": [4, 4, 4, 3, 3, 3, 2, 2, 2],
        "dilations": [1, 2, 8, 12, 16, 12, 8, 2, 1],
    }
    model = TestNet(**dilated_params)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    torch.xpu.empty_cache()
    model.to(device)
    unpad = model.pad_total + 1
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.0025, weight_decay=0.001)
    ipex.optimize(model, optimizer=optimizer)

    step_loss_weights = 1 / np.arange(2, 7)

    def multistep_loss(input, preds):
        in_L = input.shape[-1]
        out_L = preds.shape[-1]
        return torch.stack(
            [
                lw
                * F.huber_loss(
                    input[..., 1 + in_L - out_L + step : in_L],
                    preds[..., step, : out_L - step - 1],
                )
                for step, lw in enumerate(step_loss_weights)
            ]
        ).sum()

    n_epochs = 20

    losses = []
    losses_verbose = []

    model.train()
    optimizer.train()
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_input, batch_target in tqdm(ds_full_loader):
            optimizer.zero_grad()
            batch = batch_input.to(device)
            outputs = model(batch)
            loss = multistep_loss(batch_target.to(device), outputs)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        losses_verbose.append(epoch_losses)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

    model.eval()
    optimizer.eval()

    model.cpu()
    torch.save(model.state_dict(), "./tmpmodel4.pt")

    # x = torch.rand(1, 2, 249)
    # torch.onnx.export(model, x, "aaaaa4.onnx") # can't load onnx module for some reason? works in jupyter
