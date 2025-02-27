# LabletGeneralizationBenchmark

We propose a simple generalization benchmark with various systematic 
out-of-distribution test splits (composition, interpolation and extrapolation). 
This procedure is visualized in the figure below. 


![Dataset Splits](./assets/dataset_splits.png)
Fig. 1: In the four scatter plots, we see various splits along the generative
 factors of variations for the dSprites dataset. The axes correspond to
   factors of variation in the data, i.e., *scale* as visualized for
   extrapolation on the right. 

### Datasets
We consider the dSprites, Shapes3D and MPI3D-Real dataset. The splits
 corresponding to random, composition, interpolation and extrapolation can be
  found at _placeholder_. 

### Training
In this benchmark, we allow for a wide variety of modelling approaches and also
 leveraging external data. 
For instance, various types of supervision are allowed from unsupervised,
weakly-supervised, supervised to transfer-learning.
However, the test set should remain untouched and can **only** be used for
 evaluation. 
 
### Evaluation
The random, composition and interpolation splits can be used for
 hyperparameter tuning. The final evaluation and ranking can be done on the
  extrapolation setting. Please submit a pull request if you beat the state-of
  -the-art on extrapolation. 
 
Evaluating your model on this benchmark can be done with as little as 3 lines
 of code:

```python
import lablet_generalization_benchmark as lgb
import numpy as np


def model_fn(images: np.ndarray)->np.ndarray:
    # integrate your tensorflow, pytorch, jax model here
    predictions = model(images)
    return predictions

dataloader = lgb.load_dataset('shapes3d', 'extrapolation', mode='test')
# get dictionary of r2 and mse per factor
score = lgb.evaluate_model(model_fn, dataloader)  

```
We use the R2 metric for evaluation and ranking models. 


## Shapes3D Leaderboard

|                Method               |  Reference  | R2 score Extrapolation|
|-------------------------------|------------------------------------------------------------------------|:-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  
| PlaceHolder2 | placeholder |   --% |   --% | 
| PlaceHolder3 | placeholder |   --% |   --% |  
| PlaceHolder4 | placeholder |   --% |   --% | 

## dSprites Leaderboard

|                Method               |  Reference  | R2 score Extrapolation|
|-------------------------------|------------------------------------------------------------------------|:-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  
| PlaceHolder2 | placeholder |   --% |   --% | 
| PlaceHolder3 | placeholder |   --% |   --% |  
| PlaceHolder4 | placeholder |   --% |   --% | 

## MPI3D Leaderboard

|                Method               |  Reference  | R2 score Extrapolation|
|-------------------------------|------------------------------------------------------------------------|:-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  
| PlaceHolder2 | placeholder |   --% |   --% | 
| PlaceHolder3 | placeholder |   --% |   --% |  
| PlaceHolder4 | placeholder |   --% |   --% | 


## Citation

Please cite our paper at _bibtex_placeholder_. 
