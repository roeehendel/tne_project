from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import IGNORE_INDEX
from data_loading.tne_data_paths import TRAIN_DATASET
from data_loading.tne_dataset import TNEDataset
from models.architecture_configurations import DEFAULT_ARCHITECTURE_CONFIGURATION
from models.tne_model import TNEModel


def create_model():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    dataset = TNEDataset(TRAIN_DATASET, tokenizer, max_length=512, max_nps=60, ignore_index=IGNORE_INDEX)
    dataloader = DataLoader(dataset, batch_size=2)

    model = TNEModel(
        architecture_config=DEFAULT_ARCHITECTURE_CONFIGURATION
    )

    batch = next(iter(dataloader))

    model.forward(batch['inputs'])


if __name__ == '__main__':
    create_model()
