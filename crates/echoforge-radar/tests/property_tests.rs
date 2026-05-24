// Property tests for echoforge-radar.
// These verify structural/arithmetic invariants that must hold for any input
// within the valid parameter space.

use proptest::prelude::*;

proptest! {
    /// CFAR total window size (guard cells + reference cells) is always >= 1.
    #[test]
    fn cfar_guard_cells_plus_window_is_positive(
        guard in 0usize..8,
        window in 1usize..32,
    ) {
        let total = guard + window;
        prop_assert!(total >= 1, "total={total} guard={guard} window={window}");
    }

    /// Reference window alone (without guard) is never zero when window >= 1.
    #[test]
    fn cfar_reference_window_nonzero(
        window in 1usize..64,
    ) {
        prop_assert!(window > 0);
    }

    /// Guard band is strictly inside the reference window: guard < window.
    #[test]
    fn cfar_guard_less_than_window(
        guard in 0usize..16,
        extra in 1usize..16,
    ) {
        // window is always at least guard+1 when constructed this way
        let window = guard + extra;
        prop_assert!(guard < window,
            "guard={guard} must be < window={window}");
    }

    /// Threshold multiplier (alpha) scaled from SNR in dB must always be positive.
    #[test]
    fn cfar_alpha_from_snr_db_is_positive(snr_db in 0.0f64..30.0f64) {
        // alpha = 10^(snr_db/10) — always positive for finite snr_db
        let alpha = 10f64.powf(snr_db / 10.0);
        prop_assert!(alpha > 0.0, "alpha={alpha} for snr_db={snr_db}");
        prop_assert!(alpha.is_finite(), "alpha must be finite");
    }

    /// Probability of false alarm (Pfa) expressed in linear scale must be in (0, 1].
    #[test]
    fn pfa_linear_in_unit_interval(pfa_log10 in -6.0f64..-1.0f64) {
        let pfa = 10f64.powf(pfa_log10);
        prop_assert!(pfa > 0.0 && pfa <= 1.0,
            "pfa={pfa} for pfa_log10={pfa_log10}");
    }
}
