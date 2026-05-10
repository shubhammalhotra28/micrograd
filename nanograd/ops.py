"""
Phase 3 — vector ops on top of Value.

These take a list of Value objects (one per class / position) and produce either
another list (softmax) or a single scalar Value (cross_entropy).

Both ops are built entirely out of Value methods (exp, log, +, *, /, etc.),
so autograd flows through them automatically — no _backward to write here.
"""

from nanograd.engine import Value


def softmax(values):
    """
    Convert a list of Value logits into a list of Value probabilities that sum to 1.

    Formula: softmax(x_i) = e^{x_i} / sum_j(e^{x_j})

    Steps to implement:
      1. exps  = list of v.exp() for each v in values
      2. total = sum of all exps  (use sum(exps, Value(0.0)) so the start is a Value)
      3. return [e / total for e in exps]

    Optional numerical-stability upgrade (do AFTER the naive version passes the test):
      Subtract max(v.data for v in values) from each v before the .exp() call.
      Mathematically identical, prevents overflow when logits are large.
    """
    # TODO (Phase 3):
    exp = [v.exp() for v in values]
    total = sum(exp, Value(0.0))
    return [e / total for e in exp]
    

def cross_entropy(logits, target_idx):
    """
    Cross-entropy loss for a single example.

    Formula: loss = -log(softmax(logits)[target_idx])

    Inputs:
      logits      -- list of Value objects (model outputs, raw scores)
      target_idx  -- int, the index of the correct class

    Returns:
      a single Value representing the loss

    Steps to implement (naive version):
      1. probs = softmax(logits)
      2. return -(probs[target_idx].log())
    """
    
    probs = softmax(logits)
    return -(probs[target_idx].log())