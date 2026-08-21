def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    # Write code here
    true_pos = sum(actual == predicted for actual,predicted in zip(y_true,y_pred))
    errors = len(y_true) - true_pos
    f1 = 2*true_pos / ((2*true_pos)+(2*errors))

    return float(f1)
    
    pass