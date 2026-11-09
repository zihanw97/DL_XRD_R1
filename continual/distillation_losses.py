"""Feature retention for replay-based continual learning."""
import math
from torch.nn import functional as F


def feature_distillation(student, teacher, feature_scale=1.):
    """Raw coordinate retention; teacher gradients are always disabled."""
    if student.ndim != 2 or student.shape != teacher.shape or len(student) == 0:
        raise ValueError("Feature KD requires paired nonempty pre-head matrices")
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise ValueError("feature_scale must be positive")
    return F.mse_loss(student, teacher.detach()) / feature_scale ** 2
