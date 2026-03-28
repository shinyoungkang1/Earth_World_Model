"""Loss functions for EWM training."""

from ewm.losses.sigreg import (
    adaptive_lambda,
    cramer_wold_sigreg,
    cross_correlation_loss,
    cross_covariance_loss,
    epps_pulley_statistic,
    vicreg_regularization,
)

__all__ = [
    "cramer_wold_sigreg",
    "cross_correlation_loss",
    "cross_covariance_loss",
    "adaptive_lambda",
    "epps_pulley_statistic",
    "vicreg_regularization",
]
