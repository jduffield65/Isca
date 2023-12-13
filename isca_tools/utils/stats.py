import numpy as np
import scipy.stats
from typing import Union


def z_score_from_confidence_interval(confidence: Union[np.array, float]) -> Union[np.array, float]:
    """
    Given a confidence interval, this returns the z-score, $Z$. Similar to obtaining z-score from P-value with a double
    tail distribution.

    If variable, $x$, has standard deviation, $\sigma$, $x \pm Z \sigma$ is the value of $x$ with the
    desired uncertainty.

    Args:
        confidence: The confidence interval (between 0 and 1) required,
            e.g. 0.95 for 95% confidence, will return a z-score of 1.96.

    Returns:
        Z-score corresponding to confidence interval
    """
    return scipy.stats.norm.ppf(1-(1-confidence)/2)
