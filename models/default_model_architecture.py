from data_loading.tne_dataset import NUM_PREPOSITIONS

_NP_CONTEXTUAL_EMBEDDER = {
    'attention': dict(
        type='attention',
        params={
            'cross_attention': True,
            'nhead': 12,
            'num_layers': 4,
        }
    ),
    'passthrough': dict(
        type='passthrough',
        params={}
    ),
}

_PREDICTOR = {
    'basic': dict(
        type='basic',
        params={
            'hidden_size': 128,
            'num_prepositions': NUM_PREPOSITIONS,
        }
    ),
    'attention': dict(
        type='attention',
        params={
            'hidden_size': 32,
            'num_prepositions': NUM_PREPOSITIONS,
            'nhead': 4,
            'num_layers': 2
        }
    )
}

DEFAULT_ARCHITECTURE_CONFIG = dict(
    word_embedder=dict(
        type='roberta',
        params={
            'pretrained_model_name': 'roberta-base',
            'freeze_embeddings': True,
            'num_layers_to_freeze': 8,
            'num_layers_to_reinitialize': 1
        }
    ),
    np_embedder=dict(
        type='concat',
        params={}
    ),
    np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
    anchor_complement_embedder=dict(
        type='concat',
        params={}
    ),
    predictor=_PREDICTOR['attention']
)
