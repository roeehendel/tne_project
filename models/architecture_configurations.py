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

CONFIGURATIONS = {}

# the next 12 configs are adding to the base model each 'advanced' modal.
for word_embedder_type in ['roberta-base', 'spanbert-base']:
    CONFIGURATIONS.update({
        f'basic-{word_embedder_type}': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'np-embedder-attention-{word_embedder_type}': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['attention'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'coref-loss-{word_embedder_type}': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['basic'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'np_contextual-{word_embedder_type}': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['basic'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'anchor_complement-multiplicative-spanbert': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
            predictor=_PREDICTOR['basic']
        ),
        f'bias-predictor-{word_embedder_type}': dict(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic-bias-init']
        )
    })

# all next configs starts with all the advanced modals, at each
CONFIGURATIONS.update({
    'advanced-roberta-base': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    ),
    'advanced-roberta-base-but-np-embedder': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['concat'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-coref-predictor': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['none'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-np-contextual-embedder': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-anchor-complement-embedder': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
        predictor=_PREDICTOR['basic-bias-init']
    ),
    'advanced-roberta-base-but-predictor': dict(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic']
    ),
})
