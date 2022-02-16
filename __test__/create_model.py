from config import IGNORE_INDEX
from models.default_model_architecture import DEFAULT_ARCHITECTURE_CONFIG
from models.tne_model import TNEModel


def create_model():
    model = TNEModel(
        ignore_index=IGNORE_INDEX,
        learning_rate=1e-4,
        loss_weight_power=0.2,
        architecture_config=DEFAULT_ARCHITECTURE_CONFIG
    )


if __name__ == '__main__':
    create_model()
