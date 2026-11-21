import torch
from spacy_alignments import get_alignments
from torch.nn.utils.rnn import pad_sequence

MODEL_INPUT_KEYS = ("input_ids", "token_type_ids", "attention_mask")


def tokenize_text_and_target(tokenizer, text_widx, target_widx, vec_type):
    inputs = tokenizer(text_widx)
    text_tidx = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

    # 単語 -> サブワード位置の対応（[CLS] の分 +1）。対応がない単語は直前の単語に揃える
    alignments, previous_char_idx_list = [], [1]
    for char_idx_list in get_alignments(text_widx.split(), text_tidx[1:-1])[0]:
        if len(char_idx_list) == 0:
            alignments.append(previous_char_idx_list)
        else:
            char_idx_list = [c + 1 for c in char_idx_list]
            alignments.append(char_idx_list)
            previous_char_idx_list = char_idx_list

    target_tidx = alignments[target_widx][0]
    if vec_type == "mask":
        # 対象語を 1 つの [MASK] に置き換える
        inputs["input_ids"][target_tidx] = tokenizer.mask_token_id
        for _ in alignments[target_widx][1:]:
            for k in inputs.keys():
                del inputs[k][target_tidx + 1]
    inputs["target_tidx"] = target_tidx
    return inputs


def make_inputs(df, tokenizer, vec_type):
    return [
        tokenize_text_and_target(tokenizer, text_widx, target_widx, vec_type)
        for text_widx, target_widx in zip(df["text_widx"], df["target_widx"])
    ]


def collate_fn(batch):
    output_dict = {
        k: pad_sequence([torch.LongTensor(b[k]) for b in batch], batch_first=True)
        for k in MODEL_INPUT_KEYS
        if k in batch[0]
    }
    output_dict["target_tidx"] = torch.LongTensor([b["target_tidx"] for b in batch])
    return output_dict
