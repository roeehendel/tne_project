from abc import abstractmethod

from models.modules.tne_base_module import TNEBaseModule


class BaseWordEmbedder(TNEBaseModule):
    @property
    @abstractmethod
    def tokenizer(self):
        pass
