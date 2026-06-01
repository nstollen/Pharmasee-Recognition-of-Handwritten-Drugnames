import torch
import numpy as np


def ctc_greedy_decode(pred, char_list):
    # pred shape: (seq_len, num_classes)
    blank = len(char_list)
    pred_indices = pred.argmax(dim=-1).cpu().numpy()
    decoded = []
    prev = None
    for idx in pred_indices:
        if idx != prev and idx != blank:
            decoded.append(char_list[idx])
        prev = idx
    return "".join(decoded)


def ctc_beam_search_decode(pred, char_list, beam_width=3):
    # pred shape: (seq_len, num_classes+1)
    blank = len(char_list)
    pred_np = pred.cpu().numpy()
    T = pred_np.shape[0]

    beams = [("", 1.0)]

    for t in range(T):
        candidates = {}
        for seq, score in beams:
            for c in range(len(char_list) + 1):
                prob = float(pred_np[t][c])
                if prob < 1e-6:
                    continue
                if c == blank:
                    new_seq = seq
                else:
                    char = char_list[c]
                    if seq and seq[-1] == char:
                        new_seq = seq
                    else:
                        new_seq = seq + char
                new_score = score * prob
                if new_seq in candidates:
                    candidates[new_seq] += new_score
                else:
                    candidates[new_seq] = new_score

        beams = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:beam_width]

    return beams