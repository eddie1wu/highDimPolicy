from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 369

    dim_max: int = 300
    n_train: int = 150
    n_test: int = 300
    n_rep: int = 200

    rct_probability: float = 0.5

    ### Random ReLU config
    X_dim_max: int = 50

    ### Misspecified models
    degree: int = 5


