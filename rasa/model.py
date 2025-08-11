import torch
import torch.nn as nn


class RASAHead(nn.Module):
    def __init__(self, input_dim, n_pos_layers, pos_out_dim=2):
        super().__init__()
        
        self.pos_out_act_layer = nn.Sigmoid()
        self.pos_out_dim = pos_out_dim
        self.n_pos_layers = n_pos_layers
        self.input_dim = input_dim

        self.pos_pred = nn.Linear(
            self.input_dim,
            self.pos_out_dim,
            bias=False)
        self.pre_pos_layers = torch.nn.ModuleList(
            [nn.Linear(self.input_dim, self.pos_out_dim, bias=False) for i in range(self.n_pos_layers)]
        )

    def forward(self, x, use_pos_pred=True, return_pos_info=False):
        """
        Forward method for the RASAHead.
        Args:
            x: The input tensor of shape (B, N, D) where B is the batch size,
            N is the number of patches, and D is the dimension of the patch encodings.
        Returns:
            y: The output tensor of shape (B, N, D) with the positional information
            removed after using self.pre_pos_layers linear layers iteratively.
        """
        if self.n_pos_layers > 0:
            for _, l in enumerate(self.pre_pos_layers):
                x_pos, x = self.decompose_pos_2D(x, ll_weight=l.weight)  # the x will be the input to the next layer

        if use_pos_pred == True:
            # Use the last layer to remove the positional information
            x_pos, x = self.decompose_pos_2D(x, ll_weight=self.pos_pred.weight)

        if return_pos_info:
            return x_pos, x
        return x

    def forward_pos_pred(self, x):
        y = self.pos_pred(x)
        y = self.pos_out_act_layer(y)
        return y
     
    @staticmethod
    def decompose_pos_2D(x, ll_weight) -> torch.Tensor | torch.Tensor:
        bs = x.shape[0]
        ps = x.shape[1]
        pos_vr = ll_weight[0]  # shape: (D)
        pos_vc = ll_weight[1]  # shape: (D)

        # For 3-dimensional we could use the cross-product that gives us the normal vector
        # perpendicular to the plane. For n-dimensional vectors, the logic remains the same,
        # but we need to account for the dimensionality when calculating the normal vector.
        # In n-dimensions, the plane defined by two vectors A and B does not have a single
        # normal vector but rather a normal subspace. For this case, we compute the projection
        # of v onto the subspace spanned by A and B using the Gram-Schmidt process.

        # Step 1: Normalize the vectors pos_vr and pos_vc
        pos_vr = pos_vr / torch.norm(pos_vr)
        pos_vc = pos_vc / torch.norm(pos_vc)

        # Step 2: Orthogonalize pos_vc with respect to pos_vr using Gram-Schmidt
        pos_vc_orth = pos_vc - torch.dot(pos_vc, pos_vr) * pos_vr
        pos_vc_orth = pos_vc_orth / torch.norm(pos_vc_orth)  # Normalize the orthogonalized pos_vc

        # Step 3: Project x onto pos_vr and pos_vc_orth
        x_proj_pos_vr = (x * pos_vr.repeat(bs, ps, 1)).sum(dim=-1).unsqueeze(-1) * pos_vr.repeat(bs, ps, 1)
        x_proj_pos_vc_orth = (x * pos_vc_orth.repeat(bs, ps, 1)).sum(dim=-1).unsqueeze(-1) * pos_vc_orth.repeat(bs, ps, 1)

        # Step 4: Compute the projection of x onto the plane spanned by pos_vr and pos_vc
        x_pos = x_proj_pos_vr + x_proj_pos_vc_orth

        # Step 5: Remove the positional information from the vectors
        no_pos_x = x - x_pos

        return x_pos, no_pos_x


class RASAModel(nn.Module):
    def __init__(self, config, encoder : nn.Module):
        super().__init__()

        self.encoder = encoder

        # TODO: Add parsing of vit size 192, 384, 768, 1024, 1280
        embed_dim = 384
        self.head = RASAHead(embed_dim, config.n_pos_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x