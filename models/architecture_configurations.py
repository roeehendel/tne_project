from collections import OrderedDict

from models.tne_model import TNEArchitectureConfiguration

_WORD_EMBEDDER_PARAMS = {
    'freeze_embeddings': True,
    'num_layers_to_freeze': 8,
    'num_layers_to_reinitialize': 0,
    'lexical_dropout': 0.2
}

_WORD_EMBEDDER = {
    'roberta-base': dict(
        type='BaseWordEmbedder',
        params={
            'pretrained_model_name': 'roberta-base',
            **_WORD_EMBEDDER_PARAMS
        }
    ),
    'roberta-large': dict(
        type='BaseWordEmbedder',
        params={
            'pretrained_model_name': 'roberta-large',
            **_WORD_EMBEDDER_PARAMS
        }
    ),
    'spanbert-base': dict(
        type='BaseWordEmbedder',
        params={
            'pretrained_model_name': 'SpanBERT/spanbert-base-cased',
            **_WORD_EMBEDDER_PARAMS
        }
    ),
}

_NP_EMBEDDER = {
    'concat': dict(
        type='ConcatNPEmbedder',
        params={}
    ),
    'attention': dict(
        type='AttentionNPEmbedder',
        params={}
    )
}

_COREF_PREDICTOR = {
    # TODO: make this configuration work
    'none': dict(
        type='NoneCorefPredictor',
        params={}
    ),
    'basic': dict(
        type='CorefPredictor',
        params={}
    )
}

_NP_CONTEXTUAL_EMBEDDER = {
    'passthrough': dict(
        type='PassthroughNPContextualEmbedder',
        params={}
    ),
    'coref': dict(
        type='CorefNPContextualEmbedder',
        params={}
    ),
}

_ANCHOR_COMPLEMENT_EMBEDDER = {
    'concat': dict(
        type='ConcatAnchorComplementEmbedder',
        params={}
    ),
    'multiplicative': dict(
        type='MultiplicativeAnchorComplementEmbedder',
        params={}
    ),
}

_PREDICTOR = {
    'basic': dict(
        type='BasicPredictor',
        params={'initialize_bias': False}
    ),
    'basic-bias-init': dict(
        type='BasicPredictor',
        params={'initialize_bias': True}
    ),
}

ARCHITECTURE_CONFIGURATIONS = OrderedDict()

# the next 12 configs are adding to the base model each 'advanced' modal.
for word_embedder_type in ['roberta-base', 'spanbert-base']:
    ARCHITECTURE_CONFIGURATIONS.update({
        f'basic-{word_embedder_type}': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'basic-{word_embedder_type}-np-embedder': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['attention'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'basic-{word_embedder_type}-coref-loss': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['basic'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'basic-{word_embedder_type}-np-contextual': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['basic'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic']
        ),
        f'basic-{word_embedder_type}-anchor-complement': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
            predictor=_PREDICTOR['basic']
        ),
        f'basic-{word_embedder_type}-predictor': TNEArchitectureConfiguration(
            word_embedder=_WORD_EMBEDDER[f'{word_embedder_type}'],
            np_embedder=_NP_EMBEDDER['concat'],
            coref_predictor=_COREF_PREDICTOR['none'],
            np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
            anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
            predictor=_PREDICTOR['basic-bias-init']
        )
    })

# all next configs starts with all the advanced modals, at each
ARCHITECTURE_CONFIGURATIONS.update({
    'advanced-roberta-base': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    ),
    'advanced-roberta-base-but-np-embedder': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['concat'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-coref-loss': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['none'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-np-contextual': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
    ,
    'advanced-roberta-base-but-anchor-complement': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
        predictor=_PREDICTOR['basic-bias-init']
    ),
    'advanced-roberta-base-but-predictor': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-base'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic']
    ),
})

ARCHITECTURE_CONFIGURATIONS.update({
    'basic-roberta-large': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-large'],
        np_embedder=_NP_EMBEDDER['concat'],
        coref_predictor=_COREF_PREDICTOR['none'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['passthrough'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['concat'],
        predictor=_PREDICTOR['basic']
    ),
    'advanced-roberta-large': TNEArchitectureConfiguration(
        word_embedder=_WORD_EMBEDDER['roberta-large'],
        np_embedder=_NP_EMBEDDER['attention'],
        coref_predictor=_COREF_PREDICTOR['basic'],
        np_contextual_embedder=_NP_CONTEXTUAL_EMBEDDER['coref'],
        anchor_complement_embedder=_ANCHOR_COMPLEMENT_EMBEDDER['multiplicative'],
        predictor=_PREDICTOR['basic-bias-init']
    )
})
