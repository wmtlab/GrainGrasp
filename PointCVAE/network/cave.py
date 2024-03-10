import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(
        self,
        encoder_layer_sizes,
        latent_size,
        decoder_layer_sizes,
        conditional=True,
        condition_size=1088,
    ):
        super().__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size
        )
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size
        )

    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)  # (B,latent_size),(B,1088,N)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        # batch_size = n
        z = torch.randn([n, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size  # emb_dim+576

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Conv1d(in_size, out_size, 1)
            )
            self.MLP.add_module(name="B{:d}".format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        # print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=1)  # [B, emb_dim+576,N]

        x = self.MLP(x)
        x = torch.max(x, 2, keepdim=True)[0].squeeze(-1)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Conv1d(in_size, out_size, 1)
            )
            if i + 1 < len(layer_sizes):

                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c):

        N = c.shape[-1]
        if self.conditional:
            z = z.unsqueeze(-1).repeat(1, 1, N)
            z = torch.cat((z, c), dim=1)
        x = self.MLP(z)

        return x


if __name__ == "__main__":
    pass
