#!/bin/bash

RHO_MIN=0.1
RHO_DECAY=0.9
RHO_CONST=50.0
SIGMA=0.05
MASK_LIST=(
    "obs_0.05"
    "obs_0.10"
    "irr"
)


for mask in "${MASK_LIST[@]}"; do
    python posterior_sample_highres.py \
        gpu=0 \
        num_runs=5 \
        add_exp_name='' \
        +data=marseille \
        data.mask_name=$mask \
        +task=completion \
        task.operator.img_dim=2048 \
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
        sampler.decomposition.init_rank=500 \
        sampler.decomposition.num_gibbs_iters=1 \
        sampler.decomposition.use_patch=false \
        sampler.decomposition.orig_shape=[3,2048,2048] \
        sampler.decomposition.shape=[3,64,32,64,32]
done