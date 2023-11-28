from typing import List, Optional, Tuple
import numpy as np
import random
from collections import Counter

instructions = {
    "match": [
        "The system identifies melodic patterns in various songs and compares them to determine if they originate from the same track. If the patterns match, the response is 'match'; otherwise, it's 'mismatch'.", 
        "Based on the two audio signals provided, determine if they share a common origin. If the signals are from the same source, respond with 'match'; otherwise, reply with 'mismatch'.", 
        "Assess the similarity between the two audio signals and determine if they stem from the same original recording. If the signals are identical, respond with 'match'; otherwise, indicate 'mismatch'.", 
        "Evaluate the similarity between the two audio signals and determine if they share a common origin. If the signals are from the same source, respond with 'match'; otherwise, indicate 'mismatch'.", 
        "Determine the degree of similarity between the two audio signals and decide if they originated from the same source. If the signals are identical, respond with 'match'; otherwise, indicate 'mismatch'.", 
        "Based on the comparison of the two audio signals, determine if they were recorded from the same source. If the signals are identical, respond with 'match'; otherwise, indicate 'mismatch'.", 
        "Determine if the two audio signals share a common origin. If yes, respond with 'match', otherwise, answer with 'mismatch'.", 
        "Compare the two audio signals and decide if they come from the same source. Respond with 'match' if they do, and 'mismatch' if they don't.", 
        "Evaluate the similarity between the two audio signals and determine if they were recorded from the same source. Answer with 'match' for identical signals, and 'mismatch' for non-identical signals.", 
        "Assess the correspondence between the two audio signals and decide if they stem from the same original recording. Respond with 'match' for matching signals, and 'mismatch' for non-matching signals.", 
        "Investigate the similarity between the two audio signals and determine if they share a common ancestry. Answer with 'match' if they do, and 'mismatch' if they don't.", 
        "Examine the two audio signals and determine if they exhibit sufficient similarity to suggest a shared origin. If the signals are identical, respond with 'match'; otherwise, indicate 'mismatch'.", 
        "Analyze the two audio signals below and determine if they belong to the same musical composition. If the signals are from the same song, respond with 'match'; otherwise, indicate 'mismatch'.", 
    ]
}

class BaseModel:
    def __init__(self):
        pass

    def forward(
        self,
        speech_inputs: List[Tuple[np.ndarray, int]],
        text_inputs: List[str],
        instr: str,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        The forward function is designed to infer a single example (i.e., batch_size = 1).

        Input arguments:
            speech_inputs: A list of utterances. Each utterance is represented as a NumPy array, accompanied by its sampling rate.
            text_inputs: A list of texts. Each text is of type "str".
            instr: The instruction specifying the task.
        Output arguments:
            speech_outputs: A list of utterances. Each utterance is represented as a NumPy array, accompanied by its sampling rate.
            text_outputs: A list of texts. Each text is of type "str".

        """

        # Since this is a dummy model, we just return the inputs.
        return speech_inputs, text_inputs
    
def label_project(pred, labels=None):
    candidates, meta = [], {}
    for label in labels:
        if label.lower() in pred.lower():
            candidates.append(label)
    if len(candidates) == 0:
        pred = random.sample(labels, 1)[0]
    else: pred = random.sample(candidates, 1)[0]
    meta["n_cand"] = len(candidates)
    return pred, meta

def ensemble(preds, weight=None):
    mode = "max"
    meta = {}
    assert mode in ["max", "sample"]
    stats = Counter(preds)
    max_num = max(stats.values())
    meta["max_vote"] = max_num
    if mode == 'max':
        labels = [k for k, v in stats.items() if v == max_num]
        return random.sample(labels, 1)[0], meta

    elif mode == "sample":
        return random.sample(preds, 1)[0], meta