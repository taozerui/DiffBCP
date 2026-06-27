#!/bin/bash

RHO_MIN=0.3
RHO_DECAY=0.9
RHO_CONST=100.0
SIGMA=0.05
MASK_LIST=(
    "random_mask_obs01"
    "random_mask_obs03"
    "stripe_mask_row03_col03"
    "irregular_mask_area50-70_brush2-6"
)

for mask in "${MASK_LIST[@]}"; do
    python posterior_sample.py \
        gpu=0 \
        add_exp_name='' \
        +data=imagenet \
        data.mask_name=$mask \
        +task=completion \
        task.noise.sigma=$SIGMA \
        +model=edm_unet_adm_dps_imagenet \
        +sampler=pnp_edm \
        sampler.mode=vp_sde \
        sampler.num_iters=100 \
        sampler.use_tau_to_anneal=true \
        sampler.anneal_const=$RHO_CONST \
        sampler.rho=10 \
        sampler.rho_decay_rate=$RHO_DECAY \
        sampler.rho_min=$RHO_MIN \
        sampler.decomposition.use=true \
        sampler.decomposition.tau_beta=1e-3 \
        sampler.decomposition.init_rank=200 \
        sampler.decomposition.num_gibbs_iters=0 \
        sampler.decomposition.use_patch=true \
        sampler.decomposition.patch_size=16 \
        sampler.decomposition.stride=8
done