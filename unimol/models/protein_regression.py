# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore

from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch



@register_model("protein_regression")
class ProteinRegressoionModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        DTA_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.tokenizer = AutoTokenizer.from_pretrained("/cto_studio/xtalpi_lab/fengbin/DrugCLIP/esm/esm2_t12_35M_UR50D")
        self.protein_model = AutoModelForMaskedLM.from_pretrained("/cto_studio/xtalpi_lab/fengbin/DrugCLIP/esm/esm2_t12_35M_UR50D")
        # self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        self.regression_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # self.cross_distance_project = NonLinearHead(
        #     args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        # )
        # self.holo_distance_project = DistanceHead(
        #     args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        # )

        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(13))
        self.logit_bias = nn.Parameter(torch.ones([1], device="cuda") * 7)

        self.protein_project = NonLinearHead(
            self.protein_model.config.hidden_size, 128, "relu"
        )

        # self.fuse_project = NonLinearHead(
        #     256, 1, "relu"
        # )

        # self.classification_head = nn.Sequential(
        #     nn.Linear(args.pocket.encoder_embed_dim + args.pocket.encoder_embed_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def get_dist_features(self, dist, et, flag):
        if flag == "mol":
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        else:
            n_node = dist.size(-1)
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            protein_sequences,
            batch_list=None,
            encode=False,
            masked_tokens=None,
            features_only=True,
            is_train=True,
            **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(
            mol_src_distance, mol_src_edge_type, "mol"
        )
        mol_outputs = self.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias
        )
        mol_encoder_rep = mol_outputs[0]
        encoder_pair_rep = mol_outputs[1]
        mol_rep = mol_encoder_rep[:, 0, :]

        inputs = self.tokenizer(protein_sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=800)

        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()
        outputs = self.protein_model(**inputs, output_hidden_states=True)
        protein_encoder_rep = outputs.hidden_states[-1]
        protein_rep = protein_encoder_rep[:, 0, :]
        # print(protein_encoder_rep.shape, mol_rep.shape)

        mol_emb = self.mol_project(mol_rep)
        mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
        protein_emb = self.protein_project(protein_rep)
        protein_emb = protein_emb / protein_emb.norm(dim=1, keepdim=True)

        if batch_list is None:
            pocket_emb = protein_emb.unsqueeze(1).repeat(1, mol_emb.shape[0], 1)
            mol_emb = mol_emb.unsqueeze(0).repeat(pocket_emb.shape[0], 1, 1)
            concat_emb = torch.concat([mol_emb, pocket_emb], dim=-1)
            pred_act = self.regression_head(concat_emb).squeeze(-1)
            return pred_act
        else:
            num_pocket = len(batch_list)
            pred_act = []
            for i in range(num_pocket):
                range_i = batch_list[i]
                mol_emb_i = mol_emb[range_i[0]: range_i[1]]
                pocket_emb_i = protein_emb[i: i+1].repeat(mol_emb_i.shape[0], 1)
                concat_emb_i = torch.concat([mol_emb_i, pocket_emb_i], dim=1)
                pred_act_i = self.regression_head(concat_emb_i)
                pred_act.append(pred_act_i)
            return torch.concat(pred_act, dim=0)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class DistanceHead(nn.Module):
    def __init__(
            self,
            heads,
            activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("protein_regression", "protein_regression")
def DTA_architecture(args):
    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)



