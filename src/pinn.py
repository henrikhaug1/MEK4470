"""Re-export all PINN components for backward compatibility.

The actual implementations live in:
  src/models.py         — model classes, save/load, L-BFGS helper
  src/backward_step.py  — backward-facing step functions
  src/taylor_green.py   — Taylor-Green vortex functions
  src/lid_cavity.py     — lid-driven cavity functions
"""

from src.models import (  # noqa: F401
    BackwardStepPINN,
    TaylorGreenPINN,
    FourierBackwardStepPINN,
    save_model,
    save_fourier_model,
    load_fourier_model,
    load_model_state,
    lbfgs_step,
)

from src.backward_step import (  # noqa: F401
    normalize_xy,
    inlet_profile_bs,
    hard_bc_ansatz_bs,
    residuals_batch_bs,
    eval_uvp_batch_bs,
    eval_uvp_batch_bs_raw,
    dudx_at_outlet_bs_raw,
    dudy_batch_bs_raw,
    residuals_batch_bs_raw,
    sample_interior_bs,
    sample_walls_bs,
    sample_inlet_bs,
    sample_outlet_bs,
)

from src.taylor_green import (  # noqa: F401
    residuals_batch,
    eval_uvp_batch,
    sample_interior,
    loss_fn_tg,
)

from src.lid_cavity import (  # noqa: F401
    hard_bc_ansatz_lc,
    residuals_batch_lc,
    eval_uvp_batch_lc,
    sample_interior_lc,
)
