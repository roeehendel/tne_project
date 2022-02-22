from config import IGNORE_INDEX
from models.architecture_configurations import DEFAULT_ARCHITECTURE_CONFIGURATION
from models.tne_model import TNEModel


def create_model():
    model = TNEModel(
        ignore_index=IGNORE_INDEX,
        learning_rate=1e-4,
        loss_weight_power=0.2,
        architecture_config=DEFAULT_ARCHITECTURE_CONFIGURATION
    )


if __name__ == '__main__':
    create_model()
