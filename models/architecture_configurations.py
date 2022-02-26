from data_loading.tne_dataset import NUM_PREPOSITIONS

_WORD_EMBEDDER = {
    'roberta-base': dict(
        type='roberta',
        params={
            'pretrained_model_name': 'roberta-base',
            'freeze_embeddings': True,
            'num_layers_to_freeze': 8,
            'num_layers_to_reinitialize': 1
        }
    ),
    'roberta-large': dict(
        type='roberta',
        params={
            'pretrained_model_name': 'roberta-large',
            'freeze_embeddings': True,
            'num_layers_to_freeze': 8,
            'num_layers_to_reinitialize': 1
        }
    ),
    'spanbert-base': dict(
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
    # TODO: make this configuration work
    'none': dict(
        type='none',
        params={}
    ),
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
            'initialize_bias': False,
        }
    ),
    'basic-bias-init': dict(
        type='basic',
        params={
            'hidden_size': 128,
            'num_prepositions': NUM_PREPOSITIONS,
            'initialize_bias': True,
        }
    ),
}

DEFAULT_ARCHITECTURE_CONFIGURATION = dict(
    word_embedder=_WORD_EMBEDDER['roberta-base'],
    np_embedder=_NP_EMBEDDER['attention'],
    coref_predictor=_COREF_PREDICTOR['basic'],
    np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
    anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
    predictor=_PREDICTOR['basic']
)
