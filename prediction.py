import json
import os

import yaml
from pytorch_lightning import Trainer

from config import CHECKPOINTS_ROOT_DIR, IGNORE_INDEX
from data_loading.tne_data_module import TNEDataModule
from models.base_tne_model import BaseTNEModel
from utils.initialization import initialize


def predict(run_name: str, data_loaders_num_workers: int, model_checkpoint_to_use: int = 0):
    initialize(0)

    run_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, run_name)

    if model_checkpoint_to_use == -1:
        checkpoint_path = os.path.join(run_dir_path, 'last.ckpt')
    else:
        best_k_models_filepath = os.path.join(run_dir_path, "best_k_models.yaml")
        with open(best_k_models_filepath, "r") as fp:
            best_k_models = yaml.safe_load(fp)

        best_k_models_tuples = [[model_path, model_score] for model_path, model_score in best_k_models.items()]
        best_k_models_tuples.sort(key=lambda x: x[1], reverse=True)
        checkpoint_path = best_k_models_tuples[model_checkpoint_to_use][0]

    model = BaseTNEModel.load_from_checkpoint(checkpoint_path=checkpoint_path, ignore_index=IGNORE_INDEX)

    tne_data_module = TNEDataModule(model.tokenizer, num_workers=data_loaders_num_workers)
    tne_data_module.prepare_data()
    tne_data_module.setup()

    trainer = Trainer(gpus=1)

    for data_split, dataloader in [('val', tne_data_module.val_dataloader()),
                                   ('test', tne_data_module.test_dataloader())]:
        trainer.test(model, dataloaders=dataloader)
        test_results = model.test_results

        with open(os.path.join(run_dir_path, f"{data_split}_results.jsonl"), "w") as output_file:
            for example_results in test_results:
                json.dump({'predicted_prepositions': example_results}, output_file)
                output_file.write('\n')


if __name__ == '__main__':
    RUN_NAME = 'tne_model_run_4'
    DATA_LOADERS_NUM_WORKERS = 0

    predict(RUN_NAME, DATA_LOADERS_NUM_WORKERS, model_checkpoint_to_use=0)
