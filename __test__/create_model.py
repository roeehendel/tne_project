from dataclasses import asdict

from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile
from torch.utils.data import DataLoader

from config import IGNORE_INDEX
from data_loading.tne_data_paths import TRAIN_DATASET
from data_loading.tne_dataset import TNEDataset
from models.architecture_configurations import ARCHITECTURE_CONFIGURATIONS
from models.tne_model import TNEModel


def create_model():
    architecture_configuration = asdict(list(ARCHITECTURE_CONFIGURATIONS.values())[10])

    model = TNEModel(architecture_configuration)
    dataset = TNEDataset(TRAIN_DATASET, model.tokenizer, max_nps=60, ignore_index=IGNORE_INDEX)
    dataloader = DataLoader(dataset, batch_size=2)

    print(dataset[0])

    lengths = [(dataset[i]['inputs']['ids'] != 0).sum() for i in range(len(dataset))]
    batch = next(iter(dataloader))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:
        with record_function("model_inference"):
            model(batch['inputs'])

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == '__main__':
    create_model()
