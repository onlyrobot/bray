import numpy as np

from .network.network import Network
from .network.action_head import ActionHead
from .network.aggregator import DenseAggregator
from .network.decoder import CategoricalDecoder, SingleSelectiveDecoder
from .network.encoder import CommonEncoder
from .network.layer import ValueApproximator

# multi head critic config
multi_head_value_config = {
    "num_heads": 2,
    "value_loss_heads": ["norm", "test"],
    "policy_loss_heads": ["norm"],
    "gamma": {
        "norm":0.99,
        "test": 0.99,
    },
    "lamb": {
        "norm":0.95,
        "test": 0.95,
    },
}

network_input_config = {
    # ----------------------------------------------------- state input -------------------------------------------------
    "atari_vec": {
        "shape": (4,),
    },

    # ----------------------------------------------------- mask -------------------------------------------------
    "main_decision_action_mask": {
        "shape": (2,),
    },
    # ----------------------------------------------------- valid action -------------------------------------------------
    # 以valid_action_, 后缀为ActionHead的name, 所有的ActionHead均需要设置valid action
    "valid_action_main_decision": {
        "shape": (1,),
    },
}

def BuildAtariNetwork():
    evaluation = False

    atari_encoder = CommonEncoder(
        input_name="atari_vec",
        output_name="atari_vec",
        hidden_layer_sizes=[]
    )


    encoders = [
        atari_encoder,
    ]

    state_aggregator = DenseAggregator(128)

    if evaluation:
        decoder_temperature = 1e-6
    else:
        decoder_temperature = 1.0


    main_decision_cmd_decoder = CategoricalDecoder(
        n=2,
        hidden_layer_sizes=[],
        temperature=decoder_temperature,
        )
    main_decision_action_head = ActionHead(
        "main_decision",
        mask_feature_name="main_decision_action_mask",
        source_feature_name=None,
        auto_regressive=False,
        dependency="origin"
    )

    action2decoder = {
        main_decision_action_head:main_decision_cmd_decoder,
    }

    # value head
    value_layer = ValueApproximator([], num_heads=multi_head_value_config['num_heads'])

    network = Network(encoders, state_aggregator, action2decoder, value_layer, multi_head_value_config)
    target_state = {
        feature: np.ones((3, *feature_config["shape"])) for feature, feature_config in network_input_config.items()
    }

    network(target_state)
    # func_network = network.to_functional(target_state)

    return network, network_input_config, multi_head_value_config
