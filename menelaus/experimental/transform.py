import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, InputLayer
from toolz import curry
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from typing import Optional, Dict, Union, List


@curry
def auto_tokenize(data, model_name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.batch_encode_plus(data, **kwargs)
    return tokens


@curry
def _hidden_state_embeddings(hidden_states, layers, use_cls):
    hs = [
        hidden_states[layer][:, 0:1, :] if use_cls else hidden_states[layer]
        for layer in layers
    ]
    hs = tf.concat(hs, axis=1)
    y = tf.reduce_mean(hs, axis=1)
    return y


class TransformerEmbedding(tf.keras.Model):
    def __init__(
        self, model_name_or_path: str, embedding_type: str, layers: List[int] = None
    ) -> None:
        """
        Extract text embeddings from transformer models.

        Parameters
        ----------
        model_name_or_path
            Name of or path to the model.
        embedding_type
            Type of embedding to extract. Needs to be one of pooler_output,
            last_hidden_state, hidden_state or hidden_state_cls.

            From the HuggingFace documentation:

            - pooler_output
                Last layer hidden-state of the first token of the sequence
                (classification token) further processed by a Linear layer and a Tanh
                activation function. The Linear layer weights are trained from the next
                sentence prediction (classification) objective during pre-training.
                This output is usually not a good summary of the semantic content of the
                input, you’re often better with averaging or pooling the sequence of
                hidden-states for the whole input sequence.
            - last_hidden_state
                Sequence of hidden-states at the output of the last layer of the model.
            - hidden_state
                Hidden states of the model at the output of each layer.
            - hidden_state_cls
                See hidden_state but use the CLS token output.
        layers
            If "hidden_state" or "hidden_state_cls" is used as embedding
            type, layers has to be a list with int's referring to the hidden layers used
            to extract the embedding.
        """
        super(TransformerEmbedding, self).__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self.model = TFAutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.embedding_type = embedding_type
        self._hs_emb = _hidden_state_embeddings(
            layers=layers, use_cls=embedding_type.endswith("cls")
        )

    def call(self, tokens: Dict[str, tf.Tensor]) -> tf.Tensor:
        output = self.model(tokens)
        if self.embedding_type == "pooler_output":
            return output.pooler_output
        elif self.embedding_type == "last_hidden_state":
            return tf.reduce_mean(output.last_hidden_state, axis=1)
        elif self.embedding_type.startswith("hidden_state"):
            attention_hidden_states = output.hidden_states[1:]
            return self._hs_emb(attention_hidden_states)
        else:
            raise ValueError(
                "embedding_type needs to be one of pooler_output, "
                "last_hidden_state, hidden_state, or hidden_state_cls."
            )


@curry
def extract_embedding(tokens, model_name, embedding_type, layers):
    te = TransformerEmbedding(
        model_name_or_path=model_name, embedding_type=embedding_type, layers=layers
    )
    embedding = te(tokens)
    return tokens, embedding, te


class _Encoder(tf.keras.Model):
    def __init__(
        self,
        input_layer: Union[tf.keras.layers.Layer, tf.keras.Model],
        mlp: Optional[tf.keras.Model] = None,
        enc_dim: Optional[int] = None,
        step_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_layer = input_layer
        if isinstance(mlp, tf.keras.Model):
            self.mlp = mlp
        elif isinstance(enc_dim, int) and isinstance(step_dim, int):
            self.mlp = tf.keras.Sequential(
                [
                    Flatten(),
                    Dense(enc_dim + 2 * step_dim, activation=tf.nn.relu),
                    Dense(enc_dim + step_dim, activation=tf.nn.relu),
                    Dense(enc_dim, activation=None),
                ]
            )
        else:
            raise ValueError(
                "Need to provide either `enc_dim` and `step_dim` or a "
                "tf.keras.Sequential or tf.keras.Model `mlp`"
            )

    def call(self, x: Union[np.ndarray, tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        x = self.input_layer(x)
        return self.mlp(x)


class UAE(tf.keras.Model):
    # copied from alibi-detect
    def __init__(
        self,
        encoder_net: Optional[tf.keras.Model] = None,
        input_layer: Optional[Union[tf.keras.layers.Layer, tf.keras.Model]] = None,
        shape: Optional[tuple] = None,
        enc_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        is_enc = isinstance(encoder_net, tf.keras.Model)
        is_enc_dim = isinstance(enc_dim, int)
        if is_enc:
            self.encoder = encoder_net
        elif not is_enc and is_enc_dim:  # set default encoder
            input_layer = (
                InputLayer(input_shape=shape) if input_layer is None else input_layer
            )
            input_dim = np.prod(shape)
            step_dim = int((input_dim - enc_dim) / 3)
            self.encoder = _Encoder(input_layer, enc_dim=enc_dim, step_dim=step_dim)
        elif not is_enc and not is_enc_dim:
            raise ValueError(
                "Need to provide either `enc_dim` or a tf.keras.Sequential"
                " or tf.keras.Model `encoder_net`."
            )

    def call(self, x: Union[np.ndarray, tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        return self.encoder(x)


@curry
def uae_reduce_dimension(input, enc_dim, seed=0, to_numpy=True):
    tf.random.set_seed(seed)
    tokens, embedding, input_layer = input
    uae = UAE(input_layer=input_layer, shape=embedding.shape, enc_dim=enc_dim)
    embedding_reduced = uae(tokens)
    if to_numpy:
        embedding_reduced = np.array(embedding_reduced)
    return embedding_reduced
