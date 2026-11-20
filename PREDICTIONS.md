# Part C predictions

- Guidance scale s used in all four cells:  (the Part B value you will use)
- New exact target in every cell: p0 × V_A′ × V_B
- Reference from Part B at s = 1: grid 0.140, mode share 0.108 (noise floor for p*: grid 0.045, mode share 0.005)

## Cell predictions

### Guide / tighten (full model, guides A′ and B)
- Since the models are trained on all equally-weighted 8 Gaussian modes, the shares needed for target distribution will exist and be more on the upper-right corner. Guidance only has to reweight them toward the right-hand modes.
- Modes: mostly 22.5°, then 67.5° and 337.5°, roughly matching the exact shares

### Guide / reverse (full model, guides A′ and B)
- Since, the models are trained on all Gaussian modes equally and is roughly symmetric, the model will have the mass the target needs and all the bumps will upper-left corner. 
- Modes: mostly 112.5° and 157.5°. 

### Baked / tighten (model trained on x1 > 0 only, guides A′ and B)
- Since, the baked model is trained on a skewed distribution and knows only the right-hand modes, which match the shares needed for new target, the needed shares will be found. 
- Modes: 22.5°, 67.5°, 337.5°; almost none at 112.5°.

### Baked / reverse (model trained on x1 > 0 only, guides A′ and B)
- Since, the baked model is trained on a skewed distributio and most of the mass is on the left, opposite to the ones needed for new target, the needed shares will NOT be found. Guidance pushes left at every step but model will pull samples back towards the modes it knows.  This is the worst cell by far. 
- Modes: few samples in proper left-hand modes. 

## Overall
- Worst cell: baked / reverse, because guidance can only reweight probability the base model