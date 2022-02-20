from data_loading.tne_dataset import NUM_PREPOSITIONS

_WORD_EMEDDER = {
    'roberta': dict(
        type='roberta',
        params={
            # 'pretrained_model_name': 'roberta-large',
            'pretrained_model_name': 'roberta-base',
            'freeze_embeddings': True,
            'num_layers_to_freeze': 8,
            'num_layers_to_reinitialize': 1
        }
    ),
    'spanbert': dict(
        type='spanbert',
        params={
            'pretrained_model_name': 'SpanBERT/spanbert-base-cased',
            'freeze_embeddings': True,
            'num_layers_to_freeze': 8,
            'num_layers_to_reinitialize': 1
        }
    ),
}

_NP_EMBEDDER = {
    'concat': dict(
        type='concat',
        params={}
    ),
    'attention': dict(
        type='attention',
        params={}
    )
}

_COREF_PREDICTOR = {
    'basic': dict(
        type='basic',
        params={}
    )
}

_NP_CONTEXTUAL_EMBEDDER = {
    'passthrough': dict(
        type='passthrough',
        params={}
    ),
    'coref': dict(
        type='coref',
        params={}
    ),
    'transformer': dict(
        type='transformer',
        params={
            'cross_attention': True,
            'nhead': 12,
            'num_layers': 4,
        }
    ),
}

_ANCHOR_COMPLEMENT_EMBEDDER = {
    'concat': dict(
        type='concat',
        params={}
    ),
    'multiplicative': dict(
        type='multiplicative',
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
    'transformer': dict(
        type='transformer',
        params={
            'hidden_size': 32,
            'num_prepositions': NUM_PREPOSITIONS,
            'nhead': 4,
            'num_layers': 2
        }
    )
}

DEFAULT_ARCHITECTURE_CONFIG = dict(
    word_embedder=_WORD_EMEDDER['roberta'],
    np_embedder=_NP_EMBEDDER['attention'],
    coref_predictor=_COREF_PREDICTOR['basic'],
    np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
    anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
    predictor=_PREDICTOR['basic']
)
