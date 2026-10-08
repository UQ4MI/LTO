# PI-sampler: Solver-free Data Generation for Operator Learning

Research code accompanying **Physics-informed sampler for solver-free data generation in operator learning**, by Wen You, Shaoqian Zhou, Dixia Fan, and Xuhui Meng.

[1D reaction–diffusion](1D_reaction_diffusion/1D_reaction_diffusion) · [2D Poisson](2D_Poisson/2D_Poisson)

## Overview

The physics-informed sampler (PI-sampler) generates paired PDE inputs and solutions by sampling a candidate solution first and applying the governing differential and boundary operators to it. This replaces repeated numerical PDE solves during **training-data generation** with random neural-field sampling and derivative evaluation.

The examples combine the sampler with a function encoder (FE) and a latent Transformer operator (LTO):

1. Sample random neural-network parameters to obtain candidate solution fields.
2. Compute the corresponding forcing terms and boundary values.
3. Train an FE to represent forcing and solution fields in a shared learned basis.
4. Train a Transformer to map forcing coefficients and boundary/geometry information to solution coefficients.
5. Reconstruct the solution using the pretrained basis; optionally refine its coefficients using a physics-informed objective.

The supplied random-network samplers use explicit analytical derivative formulas. JAX automatic differentiation is also used for training and, in the 1D example, derivatives of the learned basis. The forcing distribution is induced by the solution prior; it is not independently prescribed during sampling.

## Included examples

| Example | Files | Main features |
| --- | --- | --- |
| 1D nonlinear reaction–diffusion | `dataset.py`, `FE.ipynb`, `OL.ipynb` | Mixture of random neural-field priors, shared function encoding, Transformer prediction with Dirichlet boundary inputs, optional L-BFGS coefficient refinement |
| 2D Poisson on parameterized geometries | `model.py`, `Train_FE.ipynb`, `Train_OL.ipynb` | Random star-shaped domains, reference-disk representation, POD-assisted function encoding, Transformer conditioned on boundary coordinates and values |

This snapshot contains these two examples from Sections 4.1 and 4.2 of the manuscript. 

## Repository structure

```text
.
├── README.md
├── SSL_SciFM.pdf
├── 1D_reaction_diffusion/
│   └── 1D_reaction_diffusion/
│       ├── dataset.py
│       ├── FE.ipynb
│       └── OL.ipynb
└── 2D_Poisson/
    └── 2D_Poisson/
        ├── model.py
        ├── Train_FE.ipynb
        └── Train_OL.ipynb
```

Keep the nested directory structure when following the paths below. The notebooks import their neighboring Python modules, so their working directory must be the **inner example directory**.



## Running the 1D reaction–diffusion example

1. Open `FE.ipynb`, select the kernel, configure the device, and run the cells in order. The notebook generates training samples on the fly, trains the shared encoder and basis, evaluates reconstruction, and saves the FE states.
2. Open `OL.ipynb` in a fresh kernel. Its early cells initialize compatible FE states and restore the checkpoints produced by `FE.ipynb`.
3. **For first-time Transformer training, skip the single-line cell that restores `transformer_linear`.** That checkpoint does not exist until the operator has been trained and saved. Keep the newly initialized `state_transformer`, then continue with the forward-function and training cells. Use the restore cell only when resuming from an existing compatible checkpoint.
4. Run the in-distribution prediction and plotting cells. The final cell optionally performs 100 L-BFGS updates of the solution coefficients while keeping the trained networks fixed.

With the current `DatasetConfig`, the generated checkpoint directories are:

```text
1D_reaction_diffusion/checkpoints/all_mixture_medium_high_freq/
├── deeponet_state_xi/       # FE coefficient encoder state
├── deeponet_state_basis/    # FE basis-network state
└── transformer_linear/     # Transformer state
```

`DatasetConfig.base_dir` points to the parent of the inner code directory, so these checkpoints are written outside the notebook directory. No pretrained checkpoints are bundled. Despite their `deeponet_*` filenames, the first two checkpoints belong to the function encoder.



## Running the 2D Poisson example

1. Open `Train_FE.ipynb`, select the kernel, configure the device, and run the cells in order. It samples random domains and fields, constructs POD modes on the reference disk, and trains a shared encoder/basis for forcing and solution reconstruction.
2. Run the final save cell. It writes the configuration, FE weights, and POD modes to `./out/`.
3. Open `Train_OL.ipynb` in a fresh kernel. It loads these three files, generates training samples on the fly, and trains the Transformer using forcing coefficients and boundary/geometry information.
4. Run the save cell that writes `Transformer.msgpack`. The later synthetic evaluation cells can be inspected separately; the cells that invoke MATLAB require the additional components described below.

The generated files are:

```text
2D_Poisson/2D_Poisson/
├── out/
│   ├── GP_R.pkl             # ModelConfig
│   ├── GP_RPrior.msgpack     # FE parameters
│   └── GP_Rpod_modes.npy     # POD modes
└── Transformer.msgpack      # Transformer parameters
```

These artifacts are not bundled and must be generated by training. The Transformer training loop also contains a periodic checkpoint save; the separate final save cell writes the final state.


