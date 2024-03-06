import tensorflow as tf
from tensorflow import keras

from .utils import get_message

def check_has_method(obj, method_name):
    method_op = getattr(obj, method_name, None)
    return callable(method_op)



class Network(keras.Model):
    def __init__(self, encoders, aggregator, action2decoder, value_approximator, multi_head_value_config):
        super(Network, self).__init__()
        self._encoders = encoders
        self._aggregator = aggregator
        self._action2decoder = action2decoder
        self._value_approximator = value_approximator
        self._multi_head_value_config = multi_head_value_config

    def get_distribution_dict(self):
        return {key.name: value.distribution_fn for key, value in self._action2decoder.items()}

    def get_initial_state(self):
        # if check_has_method(self._aggregator, "get_initial_state"):
        #     return self._aggregator.get_initial_state()
        return None

    def call(self, inputs_dict, behavior_action_dict=None, seq_len=1):
        encoder_outputs_dict = {}
        encoder_embeddings_dict = {}
        tf_log_dict = dict()
        for encoder in self._encoders:
            outputs, embeddings = encoder(inputs_dict, tf_log_dict)
            encoder_outputs_dict[encoder.output_name] = outputs
            encoder_embeddings_dict[encoder.output_name] = embeddings

        encoder_outputs_list = list(encoder_outputs_dict.values())

        hidden_state = inputs_dict.get("hidden_state", None)
        aggregator_outputs, _ = self._aggregator(
            encoder_outputs_list,
            initial_state=hidden_state,
            seq_len=seq_len,
        )

        get_message(aggregator_outputs, tf_log_dict, "aggregator_outputs")
        
        decoder_embeddings_dict = {}
        decoder_logits_dict = {}
        decoder_action_dict = {}
        decoder_info_dict = dict()

        auto_regressive_embedding = aggregator_outputs

        decoder_embeddings_dict["origin"] = auto_regressive_embedding

        # tf.gather(inputs_dict["skill_target_action_mask"], decoder_action_dict["action_select_skill"], axis=1, batch_dims=1)

        for action_head, decoder in self._action2decoder.items():

            mask = inputs_dict[action_head.mask_feature_name] if action_head.mask_feature_name else None
            source_embeddings = (
                encoder_embeddings_dict[action_head.source_feature_name]
                if action_head.source_feature_name is not None
                else None
            )
            auto_regressive_embedding = (
                decoder_embeddings_dict[action_head.dependency] if action_head.dependency else auto_regressive_embedding
            )
            behavior_action = behavior_action_dict["action_" + action_head.name] if behavior_action_dict else None
            logits, action, auto_regressive_embedding = decoder(
                [auto_regressive_embedding, source_embeddings],
                action_mask=mask,
                behavior_action=behavior_action,
                auto_regressive=action_head.auto_regressive,
                action_name=action_head.name,
                decoder_info_dict=decoder_info_dict,
                tf_log_dict=tf_log_dict,
            )

            decoder_embeddings_dict[action_head.name] = auto_regressive_embedding
            decoder_logits_dict["logits_" + action_head.name] = logits
            decoder_action_dict["action_" + action_head.name] = action
            # get_message(logits, tf_log_dict, action_head.name, mask)

        value = self._value_approximator(
            aggregator_outputs
        )

        for i, head_name in  enumerate(self._multi_head_value_config['value_loss_heads']):
            head_critic = tf.gather(value, [i], axis=-1)
            get_message(head_critic, tf_log_dict, 'critic_'+head_name)

        predict_output_dict = {**decoder_logits_dict, **decoder_action_dict}
        predict_output_dict["value"] = value

        # if aggregator_state is not None:
        #     predict_output_dict["hidden_state"] = aggregator_state
        for k, v in tf_log_dict.items():
            predict_output_dict[k] = v
        return predict_output_dict

    def to_functional(self, inputs_dict):

        model_inputs_dict = {
            key: keras.layers.Input(value.shape[1:], name=key, batch_size=None) for key, value in inputs_dict.items()
        }

        return keras.Model(inputs=model_inputs_dict, outputs=self.call(model_inputs_dict))
