import math
def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Return precision at k and recall at k.
    """
    # Write code here
    relevant_set = set(relevant)
    top_K= recommended[:k]
    relevant_count = sum(items in relevant_set for items in recommended[:k])    
    precision = relevant_count/k

    recall = relevant_count/len(relevant_set)

    return [precision,recall]
    pass