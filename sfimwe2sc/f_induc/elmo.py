"""Minimal ELMo implementation for inference (no allennlp dependency).

Ported from AllenNLP v2.10.1 (https://github.com/allenai/allennlp),
licensed under the Apache License, Version 2.0:
  - allennlp/modules/elmo.py
  - allennlp/modules/elmo_lstm.py
  - allennlp/modules/lstm_cell_with_projection.py
  - allennlp/modules/encoder_base.py
  - allennlp/modules/highway.py
  - allennlp/modules/scalar_mix.py
  - allennlp/data/token_indexers/elmo_indexer.py
  - allennlp/nn/util.py

Only the code path used by `Elmo(options_file, weight_file, n, dropout=...)`
and `batch_to_ids` is kept. Training-only features (recurrent dropout,
fine-tuning, vocabulary caching, layer norm) are removed. The stateful LSTM
behaviour of the original (hidden states are carried over between calls) is
preserved so that outputs match the original implementation.
"""

import json
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

LstmState = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Character mapping (allennlp/data/token_indexers/elmo_indexer.py)
# ---------------------------------------------------------------------------
def _make_bos_eos(character: int) -> List[int]:
    char_ids = [ELMoCharacterMapper.padding_character] * (
        ELMoCharacterMapper.max_word_length
    )
    char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = ELMoCharacterMapper.end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    max_word_length = 50
    beginning_of_sentence_character = 256
    end_of_sentence_character = 257
    beginning_of_word_character = 258
    end_of_word_character = 259
    padding_character = 260

    @staticmethod
    def convert_word_to_char_ids(word: str) -> List[int]:
        mapper = ELMoCharacterMapper
        word_encoded = word.encode("utf-8", "ignore")[: mapper.max_word_length - 2]
        char_ids = [mapper.padding_character] * mapper.max_word_length
        char_ids[0] = mapper.beginning_of_word_character
        for k, chr_id in enumerate(word_encoded, start=1):
            char_ids[k] = chr_id
        char_ids[len(word_encoded) + 1] = mapper.end_of_word_character
        return [c + 1 for c in char_ids]


BOS_CHARACTERS = _make_bos_eos(ELMoCharacterMapper.beginning_of_sentence_character)
EOS_CHARACTERS = _make_bos_eos(ELMoCharacterMapper.end_of_sentence_character)


def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    """Convert tokenized sentences to a padded tensor of character ids.

    Returns a tensor of shape (batch_size, max_num_tokens, 50).
    """
    max_len = max((len(sentence) for sentence in batch), default=0)
    padding = [0] * ELMoCharacterMapper.max_word_length
    ids = [
        [ELMoCharacterMapper.convert_word_to_char_ids(w) for w in sentence]
        + [padding] * (max_len - len(sentence))
        for sentence in batch
    ]
    return torch.LongTensor(ids)


# ---------------------------------------------------------------------------
# Utilities (allennlp/nn/util.py)
# ---------------------------------------------------------------------------
def _add_sentence_boundary_token_ids(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    sentence_begin_token: torch.Tensor,
    sentence_end_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    new_shape = list(tensor.shape)
    new_shape[1] += 2
    with_boundary = tensor.new_zeros(*new_shape)
    with_boundary[:, 1:-1, :] = tensor
    begin = sentence_begin_token.detach().to(tensor.device)
    end = sentence_end_token.detach().to(tensor.device)
    for i, j in enumerate(sequence_lengths):
        with_boundary[i, 0, :] = begin
        with_boundary[i, j + 1, :] = end
    new_mask = (with_boundary > 0).sum(dim=-1) > 0
    return with_boundary, new_mask


def _remove_sentence_boundaries(
    tensor: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    new_shape = list(tensor.shape)
    new_shape[1] -= 2
    without_boundary = tensor.new_zeros(*new_shape)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            without_boundary[i, : j - 2, :] = tensor[i, 1 : j - 1, :]
    return without_boundary


def _sort_batch_by_length(
    tensor: torch.Tensor, sequence_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_lengths, restoration_indices, permutation_index


# ---------------------------------------------------------------------------
# Character encoder (allennlp/modules/elmo.py, highway.py)
# ---------------------------------------------------------------------------
class _Highway(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = torch.nn.functional.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * current_input + (1 - gate) * nonlinear_part
        return current_input


class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(self, options: Dict, weight_file: str) -> None:
        super().__init__()
        self._options = options
        cnn_options = options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            self._activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            self._activation = torch.nn.functional.relu
        else:
            raise ValueError(
                f"Unknown char_cnn activation: {cnn_options['activation']}"
            )
        self.output_dim = options["lstm"]["projection_dim"]

        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        char_embed_dim = cnn_options["embedding"]["dim"]

        with h5py.File(weight_file, "r") as fin:
            # Character embedding (index 0 is reserved for padding).
            char_embed = fin["char_embed"][...]
            weights = np.zeros(
                (char_embed.shape[0] + 1, char_embed.shape[1]), dtype="float32"
            )
            weights[1:, :] = char_embed
            self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=False
            )

            # Convolutions.
            self._convolutions = torch.nn.ModuleList()
            for i, (width, num) in enumerate(filters):
                conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True,
                )
                weight = fin["CNN"][f"W_cnn_{i}"][...]
                bias = fin["CNN"][f"b_cnn_{i}"][...]
                w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
                if w_reshaped.shape != tuple(conv.weight.shape):
                    raise ValueError(
                        f"Invalid weight file: W_cnn_{i} has shape "
                        f"{w_reshaped.shape}, expected {tuple(conv.weight.shape)}"
                    )
                conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
                conv.bias.data.copy_(torch.FloatTensor(bias))
                self._convolutions.append(conv)

            # Highway layers.
            n_highway = cnn_options["n_highway"]
            self._highways = _Highway(n_filters, n_highway)
            for k in range(n_highway):
                group = fin[f"CNN_high_{k}"]
                w_transform = np.transpose(group["W_transform"][...])
                w_carry = -1.0 * np.transpose(group["W_carry"][...])
                b_transform = group["b_transform"][...]
                b_carry = -1.0 * group["b_carry"][...]
                layer = self._highways._layers[k]
                layer.weight.data.copy_(
                    torch.FloatTensor(np.concatenate([w_transform, w_carry], 0))
                )
                layer.bias.data.copy_(
                    torch.FloatTensor(np.concatenate([b_transform, b_carry], 0))
                )

            # Projection.
            self._projection = torch.nn.Linear(n_filters, self.output_dim)
            self._projection.weight.data.copy_(
                torch.FloatTensor(np.transpose(fin["CNN_proj"]["W_proj"][...]))
            )
            self._projection.bias.data.copy_(
                torch.FloatTensor(fin["CNN_proj"]["b_proj"][...])
            )

        for param in self.parameters():
            param.requires_grad = False

        self._bos_characters = torch.from_numpy(np.array(BOS_CHARACTERS) + 1)
        self._eos_characters = torch.from_numpy(np.array(EOS_CHARACTERS) + 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (inputs > 0).sum(dim=-1) > 0
        char_ids, mask_with_bos_eos = _add_sentence_boundary_token_ids(
            inputs, mask, self._bos_characters, self._eos_characters
        )
        max_chars = self._options["char_cnn"]["max_characters_per_token"]
        embedding = torch.nn.functional.embedding(
            char_ids.view(-1, max_chars), self._char_embedding_weights
        )
        embedding = torch.transpose(embedding, 1, 2)
        convs = []
        for conv in self._convolutions:
            convolved, _ = torch.max(conv(embedding), dim=-1)
            convs.append(self._activation(convolved))
        token_embedding = torch.cat(convs, dim=-1)
        token_embedding = self._highways(token_embedding)
        token_embedding = self._projection(token_embedding)
        batch_size, sequence_length, _ = char_ids.size()
        return (
            token_embedding.view(batch_size, sequence_length, -1),
            mask_with_bos_eos,
        )


# ---------------------------------------------------------------------------
# biLSTM (allennlp/modules/lstm_cell_with_projection.py, elmo_lstm.py,
#         encoder_base.py)
# ---------------------------------------------------------------------------
class _LstmCellWithProjection(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_size: int,
        go_forward: bool,
        memory_cell_clip_value: Optional[float],
        state_projection_clip_value: Optional[float],
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.memory_cell_clip_value = memory_cell_clip_value
        self.state_projection_clip_value = state_projection_clip_value
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)
        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        batch_lengths: List[int],
        initial_state: Optional[LstmState] = None,
    ) -> Tuple[torch.Tensor, LstmState]:
        batch_size, total_timesteps = inputs.size()[:2]
        output_accumulator = inputs.new_zeros(
            batch_size, total_timesteps, self.hidden_size
        )
        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        cs = self.cell_size
        current_length_index = batch_size - 1 if self.go_forward else 0
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while (
                    current_length_index < len(batch_lengths) - 1
                    and batch_lengths[current_length_index + 1] > index
                ):
                    current_length_index += 1

            n = current_length_index + 1
            previous_memory = full_batch_previous_memory[0:n].clone()
            previous_state = full_batch_previous_state[0:n].clone()
            timestep_input = inputs[0:n, index]

            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            gates = projected_input + projected_state
            input_gate = torch.sigmoid(gates[:, 0 * cs : 1 * cs])
            forget_gate = torch.sigmoid(gates[:, 1 * cs : 2 * cs])
            memory_init = torch.tanh(gates[:, 2 * cs : 3 * cs])
            output_gate = torch.sigmoid(gates[:, 3 * cs : 4 * cs])

            memory = input_gate * memory_init + forget_gate * previous_memory
            if self.memory_cell_clip_value:
                memory = torch.clamp(
                    memory,
                    -self.memory_cell_clip_value,
                    self.memory_cell_clip_value,
                )
            timestep_output = self.state_projection(output_gate * torch.tanh(memory))
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(
                    timestep_output,
                    -self.state_projection_clip_value,
                    self.state_projection_clip_value,
                )

            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:n] = memory
            full_batch_previous_state[0:n] = timestep_output
            output_accumulator[0:n, index] = timestep_output

        final_state = (
            full_batch_previous_state.unsqueeze(0),
            full_batch_previous_memory.unsqueeze(0),
        )
        return output_accumulator, final_state


class _ElmoLstm(torch.nn.Module):
    """Stacked biLSTM with residual connections.

    Stateful: the final hidden states of each call are used as the initial
    states of the next call, exactly as in allennlp's ElmoLstm.
    """

    def __init__(self, options: Dict, weight_file: str) -> None:
        super().__init__()
        lstm_options = options["lstm"]
        if not lstm_options.get("use_skip_connections"):
            raise ValueError(
                "Only pretrained biLMs with residual connections are supported"
            )
        self.hidden_size = lstm_options["projection_dim"]
        self.cell_size = lstm_options["dim"]
        self.num_layers = lstm_options["n_layers"]
        self.forward_layers = torch.nn.ModuleList()
        self.backward_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            for layers, go_forward in [
                (self.forward_layers, True),
                (self.backward_layers, False),
            ]:
                layers.append(
                    _LstmCellWithProjection(
                        input_size=self.hidden_size,
                        hidden_size=self.hidden_size,
                        cell_size=self.cell_size,
                        go_forward=go_forward,
                        memory_cell_clip_value=lstm_options["cell_clip"],
                        state_projection_clip_value=lstm_options["proj_clip"],
                    )
                )
        self._load_weights(weight_file)
        for param in self.parameters():
            param.requires_grad = False
        self._states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _load_weights(self, weight_file: str) -> None:
        with h5py.File(weight_file, "r") as fin:
            for i_layer, lstms in enumerate(
                zip(self.forward_layers, self.backward_layers)
            ):
                for j_direction, lstm in enumerate(lstms):
                    cs = lstm.cell_size
                    dataset = fin[f"RNN_{j_direction}"]["RNN"]["MultiRNNCell"][
                        f"Cell{i_layer}"
                    ]["LSTMCell"]
                    # TensorFlow gate order (i, j, f, o) -> PyTorch (i, f, j, o)
                    tf_weights = np.transpose(dataset["W_0"][...])
                    torch_weights = tf_weights.copy()
                    torch_weights[cs : 2 * cs, :] = tf_weights[2 * cs : 3 * cs, :]
                    torch_weights[2 * cs : 3 * cs, :] = tf_weights[cs : 2 * cs, :]
                    lstm.input_linearity.weight.data.copy_(
                        torch.FloatTensor(torch_weights[:, : lstm.input_size])
                    )
                    lstm.state_linearity.weight.data.copy_(
                        torch.FloatTensor(torch_weights[:, lstm.input_size :])
                    )
                    # TensorFlow adds 1 to the forget gate bias at runtime.
                    tf_bias = dataset["B"][...]
                    tf_bias[2 * cs : 3 * cs] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[cs : 2 * cs] = tf_bias[2 * cs : 3 * cs]
                    torch_bias[2 * cs : 3 * cs] = tf_bias[cs : 2 * cs]
                    lstm.state_linearity.bias.data.copy_(torch.FloatTensor(torch_bias))
                    lstm.state_projection.weight.data.copy_(
                        torch.FloatTensor(np.transpose(dataset["W_P_0"][...]))
                    )

    def reset_states(self) -> None:
        self._states = None

    def _get_initial_states(
        self, batch_size: int, num_valid: int, sorting_indices: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            n_concat = batch_size - self._states[0].size(1)
            self._states = tuple(
                torch.cat([s, s.new_zeros(s.size(0), n_concat, s.size(2))], 1)
                for s in self._states
            )
            shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            shaped_states = tuple(s[:, :batch_size, :] for s in self._states)
        else:
            shaped_states = self._states
        return tuple(
            s.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
            for s in shaped_states
        )

    def _update_states(
        self,
        final_states: Tuple[torch.Tensor, ...],
        restoration_indices: torch.Tensor,
    ) -> None:
        new_unsorted_states = [
            s.index_select(1, restoration_indices) for s in final_states
        ]
        if self._states is None:
            self._states = tuple(s.data for s in new_unsorted_states)
            return
        current_batch_size = self._states[0].size(1)
        new_batch_size = final_states[0].size(1)
        used_new_rows_mask = [
            (s[0, :, :].sum(-1) != 0.0).float().view(1, new_batch_size, 1)
            for s in new_unsorted_states
        ]
        new_states = []
        if current_batch_size > new_batch_size:
            for old, new, used in zip(
                self._states, new_unsorted_states, used_new_rows_mask
            ):
                masked_old = old[:, :new_batch_size, :] * (1 - used)
                old[:, :new_batch_size, :] = new + masked_old
                new_states.append(old.detach())
        else:
            for old, new, used in zip(
                self._states, new_unsorted_states, used_new_rows_mask
            ):
                new += old * (1 - used)
                new_states.append(new.detach())
        self._states = tuple(new_states)

    def _lstm_forward(
        self,
        inputs: PackedSequence,
        initial_state: Optional[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if initial_state is None:
            hidden_states: List[Optional[LstmState]] = [None] * self.num_layers
        else:
            hidden_states = list(
                zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0))
            )

        padded_inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output = padded_inputs
        backward_output = padded_inputs
        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = self.forward_layers[layer_index]
            backward_layer = self.backward_layers[layer_index]
            forward_cache = forward_output
            backward_cache = backward_output

            forward_state: Optional[LstmState] = None
            backward_state: Optional[LstmState] = None
            if state is not None:
                fwd_hidden, bwd_hidden = state[0].split(self.hidden_size, 2)
                fwd_memory, bwd_memory = state[1].split(self.cell_size, 2)
                forward_state = (fwd_hidden, fwd_memory)
                backward_state = (bwd_hidden, bwd_memory)

            forward_output, forward_state = forward_layer(
                forward_output, batch_lengths, forward_state
            )
            backward_output, backward_state = backward_layer(
                backward_output, batch_lengths, backward_state
            )
            # Residual connections (skip the first layer).
            if layer_index != 0:
                forward_output += forward_cache
                backward_output += backward_cache

            sequence_outputs.append(torch.cat([forward_output, backward_output], -1))
            final_states.append(
                (
                    torch.cat([forward_state[0], backward_state[0]], -1),
                    torch.cat([forward_state[1], backward_state[1]], -1),
                )
            )

        final_hidden, final_memory = zip(*final_states)
        return torch.stack(sequence_outputs), (
            torch.cat(final_hidden, 0),
            torch.cat(final_memory, 0),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, total_sequence_length = mask.size()
        num_valid = int(torch.sum(mask[:, 0]).int().item())
        sequence_lengths = mask.sum(-1)
        (
            sorted_inputs,
            sorted_lengths,
            restoration_indices,
            sorting_indices,
        ) = _sort_batch_by_length(inputs, sequence_lengths)
        packed_input = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_lengths[:num_valid].data.tolist(),
            batch_first=True,
        )
        initial_states = self._get_initial_states(
            batch_size, num_valid, sorting_indices
        )
        stacked_output, final_states = self._lstm_forward(packed_input, initial_states)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_output.size()
        if num_valid < batch_size:
            stacked_output = torch.cat(
                [
                    stacked_output,
                    stacked_output.new_zeros(
                        num_layers,
                        batch_size - num_valid,
                        returned_timesteps,
                        encoder_dim,
                    ),
                ],
                1,
            )
            final_states = tuple(
                torch.cat(
                    [
                        s,
                        s.new_zeros(num_layers, batch_size - num_valid, s.size(-1)),
                    ],
                    1,
                )
                for s in final_states
            )
        length_difference = total_sequence_length - returned_timesteps
        if length_difference > 0:
            stacked_output = torch.cat(
                [
                    stacked_output,
                    stacked_output.new_zeros(
                        num_layers, batch_size, length_difference, encoder_dim
                    ),
                ],
                2,
            )
        self._update_states(final_states, restoration_indices)
        return stacked_output.index_select(1, restoration_indices)


# ---------------------------------------------------------------------------
# Public API (allennlp/modules/elmo.py, scalar_mix.py)
# ---------------------------------------------------------------------------
class Elmo(torch.nn.Module):
    """Inference-only ELMo compatible with `allennlp.modules.elmo.Elmo`.

    Each output representation is a scalar mix of the biLM layers. Since the
    mixing weights are not trained here, each representation is the
    softmax([0, ..., 0])-weighted (i.e. uniform) average of the layers,
    identical to allennlp's freshly-initialised ScalarMix.
    """

    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        with open(options_file, "r") as fin:
            options = json.load(fin)
        self._token_embedder = _ElmoCharacterEncoder(options, weight_file)
        self._elmo_lstm = _ElmoLstm(options, weight_file)
        self.num_layers = options["lstm"]["n_layers"] + 1
        self.num_output_representations = num_output_representations
        self._dropout = torch.nn.Dropout(p=dropout)
        # Untrained ScalarMix parameters (scalar weights = 0, gamma = 1).
        self._scalar_parameters = torch.nn.Parameter(
            torch.zeros(self.num_layers), requires_grad=False
        )
        self._gamma = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)

    def reset_states(self) -> None:
        self._elmo_lstm.reset_states()

    def _scalar_mix(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        normed_weights = torch.split(
            torch.nn.functional.softmax(self._scalar_parameters, dim=0),
            split_size_or_sections=1,
        )
        return self._gamma * sum(w * t for w, t in zip(normed_weights, tensors))

    def forward(self, inputs: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        if inputs.dim() != 3:
            raise ValueError(
                "inputs must have shape (batch_size, timesteps, 50), "
                f"got {tuple(inputs.shape)}"
            )
        type_representation, mask = self._token_embedder(inputs)
        lstm_outputs = self._elmo_lstm(type_representation, mask)
        layer_activations = [
            torch.cat([type_representation, type_representation], dim=-1)
            * mask.unsqueeze(-1)
        ] + [
            layer.squeeze(0)
            for layer in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0)
        ]
        representations = [
            self._dropout(
                _remove_sentence_boundaries(self._scalar_mix(layer_activations), mask)
            )
            for _ in range(self.num_output_representations)
        ]
        return {"elmo_representations": representations}
