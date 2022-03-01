from inspect import isclass

from models import modules
from utils.imports import import_submodules


def get_tne_modules_classes():
    classes = dict()

    for module in import_submodules(modules).values():
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute):
                classes[attribute_name] = attribute

    return classes
