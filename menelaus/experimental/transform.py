""" 
Contains transform functions, which are curried functions initialized with a certain configuration,
and called in some sequence to transform an initial batch of data into a final formatted data
representation. Applying transforms helps compare two sets of data, and convert data into a format
accepted by some ``Alarm`` type.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, InputLayer
from toolz import curry
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from typing import Optional, Dict, Union, List


@curry
def auto_tokenize(data, model_name, **kwargs):
    """
    Curried function that takes raw data (typically list of strings), model
    name, and other optional arguments, and returns the tokens created from
    using the pre-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.batch_encode_plus(data, **kwargs)
    return tokens


@curry
def _hidden_state_embeddings(hidden_states, layers, use_cls):
    """
    Curried helper function to assist with extracting embeddings from tokens.

    Ref. :cite:t:`alibi-detect`
    """
    hs = [
        hidden_states[layer][:, 0:1, :] if use_cls else hidden_states[layer]
        for layer in layers
    ]
    hs = tf.concat(hs, axis=1)
    y = tf.reduce_mean(hs, axis=1)
    return y


class TransformerEmbedding(tf.keras.Model):
    """
    Extracts texts embeddings from transformer models. Pulled directly from ``alibi-detect``.

    Ref. :cite:t:`alibi-detect`

    Attributes:
        model_name_or_path
            Name of or path to the transformer model.
        embedding_type
            Type of embedding to extract. Needs to be one of pooler_output,
            last_hidden_state, hidden_state or hidden_state_cls.
        layers
            If "hidden_state" or "hidden_state_cls" is used as embedding
            type, layers has to be a list with int's referring to the hidden layers used
            to extract the embedding.
    """

    def __init__(
        self, model_name_or_path: str, embedding_type: str, layers: List[int] = None
    ) -> None:
        """
        Args:
            model_name_or_path
                Name of or path to the transformer model.
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
        """
        Applies transformer model to tokens, then extracts embeddings from output.

        Args:
            tokens
                Dictionary output of transformer model on raw strings. For details
                on output format, see return values of ``encode_plus``, ``__call__``,
                or ``batch_encode_plus`` methods in ``transformers.BatchEncoding``.

        Returns:
            Extracted embeddings, typically as ``tf.tensor`` or ``numpy.ndarray``.
        """
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
    """
    Curried function to extract embeddings from tokens. Takes tokens,
    name of transformer embedding model, embedding type, and layers.
    Returns tokens, embedding, and embedding model.

    Ref. :cite:t:`alibi-detect`
    """
    te = TransformerEmbedding(
        model_name_or_path=model_name, embedding_type=embedding_type, layers=layers
    )
    embedding = te(tokens)
    return tokens, embedding, te


class _Encoder(tf.keras.Model):
    """
    Helper class to assist with encoding embeddings into a reduced-dimension
    output. Pulled directly from ``alibi-detect``.

    Ref. :cite:t:`alibi-detect`

    Attributes:
        input_layer
            Input layer from which new encodings will be generated.
        mlp
            Multilayer perceptron network used for dimension-reduction step.
    """

    def __init__(
        self,
        input_layer: Union[tf.keras.layers.Layer, tf.keras.Model],
        mlp: Optional[tf.keras.Model] = None,
        enc_dim: Optional[int] = None,
        step_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            input_layer
                Input layer from which new encodings will be generated.
            mlp
                Multilayer perceptron network used for dimension-reduction step.
                Default ``None``.
            enc_dim
                Desired size for final encoded output. Default ``None``.
            step_dim
                Optional step size for constructing MLP if none given. Default ``None``.
        """
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
        """
        Performs reduced-dimension encoding step on new data.

        Args:
            x
                New input batch.

        Returns
            Encoded data (processed through input layer and MLP).
        """
        x = self.input_layer(x)
        return self.mlp(x)


class UAE(tf.keras.Model):
    """
    Untrained AutoEncoder class to reduce dimension of embedding output from previous
    steps. Pulled directly from ``alibi-detect``.

    Ref. :cite:t:`alibi-detect`

    Attributes:
        encoder
            Encoder network to be used on tokens resulting in reduced-dimension
            embeddings.
    """

    def __init__(
        self,
        encoder_net: Optional[tf.keras.Model] = None,
        input_layer: Optional[Union[tf.keras.layers.Layer, tf.keras.Model]] = None,
        shape: Optional[tuple] = None,
        enc_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            encoder_net
                If this is given as a ``tf.keras.Model``, use this to obtain embeddings.
                Default ``None``.
            input_layer
                If ``encoder_net`` not given, this is used as the input layer which
                accepts tokens. Default ``None``.
            shape
                If ``encoder_net`` not given, this is the desired input shape for the
                input layer. Default ``None``.
            enc_dim
                If ``encoder_net`` not given, this is the desired encoding dimension
                for the final output. Default ``None``.
        """
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
        """
        Performs encoding step on tensors.

        Args:
            x
                New batch of tensors.

        Returns:
            Encoded and reduced-dimension output via UAE.
        """
        return self.encoder(x)


@curry
def uae_reduce_dimension(input, enc_dim, seed=0, to_numpy=True):
    """
    Curried function to reduce dimension of embedding output via Untrained
    AutoEncoder. Takes input tuple (tokens, embedding, input layer), encoding
    dimension size, optional seed. Returns reduced array (numpy or tensor).

    Ref. :cite:t:`alibi-detect`
    """
    tf.random.set_seed(seed)
    tokens, embedding, input_layer = input
    uae = UAE(input_layer=input_layer, shape=embedding.shape, enc_dim=enc_dim)
    embedding_reduced = uae(tokens)
    if to_numpy:
        embedding_reduced = np.array(embedding_reduced)
    return embedding_reduced
