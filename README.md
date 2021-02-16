# LabletGeneralizationBenchmark

** Describe LabletGeneralizationBenchmark here **

Evaluating your model on this benchmark can be done with as little as 3 lines of code:

```python
import lablet_generalization_benchmark as lgb
import numpy as np


def model_fn(images: np.ndarray)->np.ndarray:
    predictions = np.zeros(images.shape)
    # integrate your tensorflow, pytorch, jax model here
    return predictions

dataloader = lgb.load_dataset('shapes3d', 'extrapolation', mode='train')
score_dict = lgb.evaluate_model(model_fn, dataloader)

```

## Shapes3D Leaderboard

|                Method               |  Reference  | R2 score none | R2 score random | R2 score interpolation | R2 score composition | R2 score extrapolation |
|-------------------------------|------------------------------------------------------------------------|:-------:|:-------:| :-------:| :-------:| :-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder2 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder3 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder4 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |

## dSprites Leaderboard

|                Method               |  Reference  | R2 score none | R2 score random | R2 score interpolation | R2 score composition | R2 score extrapolation |
|-------------------------------|------------------------------------------------------------------------|:-------:|:-------:| :-------:| :-------:| :-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder2 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder3 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder4 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |

## MPI3D Leaderboard

|                Method               |  Reference  | R2 score none | R2 score random | R2 score interpolation | R2 score composition | R2 score extrapolation |
|-------------------------------|------------------------------------------------------------------------|:-------:|:-------:| :-------:| :-------:| :-------:|
| PlaceHolder1 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder2 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder3 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |
| PlaceHolder4 | placeholder |   --% |   --% |  --% |  --% |  --% |  --% |


## Documentation

Generated documentation for the latest released version can be accessed here:
https://devcentral.amazon.com/ac/brazil/package-master/package/go/documentation?name=LabletGeneralizationBenchmark&interface=1.0&versionSet=live

## Development

See [DEVELOPMENT.md](./DEVELOPMENT.md)
