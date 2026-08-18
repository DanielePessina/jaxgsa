# Test sweep — adjudicated deletion list

Eight auditors swept all 41 test files against one rule: **a test earns its place
only if it would fail for a real behavioural regression that no other test catches.**
Tests that must be edited whenever the source is edited assert implementation, not
behaviour.

Delete **by test name, never by line number** — line numbers shift as you go.
Do not touch any test not named here. Do not weaken a test instead of deleting it.

## Cross-cutting: boilerplate absorbed by the Wave 3/4 generic tests

Wave 4A made `on_invalid` a single code path. There is exactly one raise site
(`_core/invalid.py:232`) and two callers: `_core/entry.py:329` (the shared preamble,
used by twelve methods) and `dgsm/_analyze.py:578`.

**C1. Delete the per-method `on_invalid` name-rejection test** from: pawn, hdmr,
vkoga, kucherenko, efast, hsic, shapley, borgonovo, cleaning, optimal_transport,
pce, morris. Names vary: `test_a_bad_policy_name_is_refused`,
`test_rejects_an_unknown_policy`, `test_a_bad_on_invalid_value_is_rejected`.
**KEEP dgsm's** (`test_a_bad_policy_value_is_rejected`) — dgsm is the one method
that resolves its own policy outside the preamble. **KEEP** the generic ones in
`test_invalid_policy.py` and `test_entry_preamble.py`.

**C2. Delete the per-method "clean sample under every policy" test** from: pawn,
hdmr, vkoga, hsic, shapley, optimal_transport, pce, borgonovo. Covered by
`test_invalid_policy.py::TestCleanSample` and `test_entry_preamble.py::TestACleanSampleStaysSilent`.

**C3. Delete the per-method "dropping an X row takes its Y row" test** from: pawn,
hsic, optimal_transport, borgonovo. Covered by
`test_invalid_policy.py::test_a_bad_input_takes_its_output_with_it`.

**C4. Delete the per-method X-shape validation trio** (`test_x_wrong_ndim`,
`test_x_wrong_columns`, `test_row_mismatch`) from pawn, dgsm, hsic. These exercise
the shared given-data validator, already covered by `test_output_shapes.py`.

**C5. Delete per-method `to_dataset` schema tests** where the claim is data-var
names / dims only — `tests/data/result_dataset_schema.json` pins these for every
method and Y-rank via `test_result_schema.py::test_dataset_schema_is_unchanged`.
Keep any assertion about coordinate *values* or kwarg behaviour by folding it into
one surviving test per file.

## Per-file deletions

### tests/test_analytical.py
DELETE: `TestIshigami::test_s1_sum_leq_1`, `TestIshigami::test_st_geq_s1`,
`TestLinear::test_s1_sums_to_1`, `TestSobolG::test_st_geq_s1`,
`TestSobolG::test_analytical_indices_consistency`, `TestOakleyOHagan::test_st_geq_s1`.
Each is implied by the analytical-value oracle above it in the same class.

### tests/test_dgsm.py
DELETE: `TestMarginalVariance::test_uniform_wide`, `TestAxisConstants::test_shapes`,
`TestAxisConstants::test_uniform_values`, all four `TestSampleMC` tests (they test
`monte_carlo`, which `test_sampling.py` owns), `TestLinearDGSM::test_bracket_contains_st`,
`TestPrecomputed::test_precomputed_matches_autodiff`, `TestPrecomputed::test_singleton_pair_2d`,
`TestChunked::test_chunked_matches_unchunked` (the ragged twin covers both branches),
`TestValidation::test_sample_mc_n_zero_raises`, `TestToDataset::test_scalar_output`,
`TestToDataset::test_multi_output`, `test_single_param`,
`TestInvalidPolicyShared::test_clean_precomputed_sample_reports_nothing`.
KEEP `TestInvalidPolicyShared::test_a_bad_policy_value_is_rejected` (see C1).

### tests/test_categorical.py
DELETE: `test_categorical_labels_empty_without_categorical_params`,
`test_categorical_input_spec_typed_dict_accepted`,
`test_borgonovo_in_range_confidence_bound_is_silent`,
`test_borgonovo_default_chunk_size_respects_budget_on_imbalanced_categorical` (mirror:
recomputes the source's own budget expression via a monkeypatched private kernel),
`test_pawn_continuous_only_binning_is_unchanged` (mirror of `pawn/_analyze.py:136`),
`test_sobol_expand_outputs_round_trips_duplicate_rows` (mirror of `expand_outputs`' body),
`test_gated_methods_raise_naming_the_categorical_parameter` (now probed empirically
by `test_registry.py::test_the_categorical_claim_matches_behaviour`).
COLLAPSE: `test_borgonovo_atomic_class_delta_stays_in_range` from 7 noise levels to
`[1e-9, 1e-5, 0.1]` — the three real behaviour classes.

### tests/test_hdmr.py
DELETE: `test_s1_via_sa`, `test_s1_accuracy` (both identical to `test_st_accuracy`),
`test_repeated_calls_identical` (hdmr.analyze has no RNG — a pure function called
twice), `test_explicit_prenormalize_false_matches_default` (passes a keyword its own
default), `test_maxorder_1`, `test_maxorder_2`, `test_maxorder_3` (each recomputes the
source's combinatorial term-count) — replace all three with ONE test asserting the
literal counts `{1: 3, 2: 6, 3: 7}` for D=3, `test_emulator_reasonable`,
`test_predict_preserves_explicit_time_series_layout`, `test_term_labels`,
`test_select_and_rmse`, `test_s1_property` (mirror: `S1` IS `Sa[..., :D]`),
`test_scalar_like_lambdax`, `test_s2_multi_output_shape`, `test_to_dataset_includes_s2_s3`,
`test_st_is_not_a_conditional_variance_total`, `test_slice_chunk_size_regression_3d`
(also the known-flaky tolerance), plus C1/C2.
TRIM: in `test_shapes_1d`, drop the `_fit` private-dict assertions; keep the public
`Sa/Sb/S/ST/rmse` shape assertions.

### tests/test_hdmr_streaming.py
DELETE: `test_predict_round_trip`, `test_invalid_batch_size_raises`,
`test_explicit_batch_size_engages_streaming`, `test_default_budget_keeps_in_memory`,
`test_full_fit_bytes_formula` (mirror — its own docstring concedes the literal is
hand-derived from the source's documented breakdown),
`TestHDMRStaticDataLayout::test_values_land_on_the_right_names`.
`TestHDMRStaticDataLayout::test_field_order_is_unchanged` — see PRODUCTION P1; delete
only after the source fix lands.

### tests/test_vkoga.py
DELETE: `test_cv_rmse_is_none_when_both_hyperparameters_are_fixed`,
`test_categorical_problem_raises`, `test_copula_fit_with_ties_matches_spearmanr` (tests
`_core.copula`, not vkoga; `test_copula.py` covers it tighter),
`test_materially_indefinite_correlation_is_rejected`, `test_to_dataset_schema`,
`test_to_dataset_time_series_dims`, `test_is_correlated_agrees_with_problem_classification`
(mirror: both sides call the same helper), `test_float32_emits_precision_warning` (ten
other tests already wrap calls in the same `pytest.warns`),
`test_cross_validation_path_resolves_gamma`, `test_estimator_uses_the_factor_on_the_plan`,
plus C1/C2.
TRIM: `test_correlation_matrix_validation_errors` — keep only the `(3,3)` shape case.

### tests/test_pce.py
DELETE: `test_s1_x3_near_zero`, `test_s1_sums_to_one`, `test_s2_shape`, `test_loo_finite`,
`test_y_ndim_4_raises`, `test_predict_1d_x_raises`, `test_predict_x_column_mismatch_raises`,
`test_dataset_type`, `test_dataset_has_required_vars`, `test_dataset_s1_dims`,
`test_dataset_s2_dims`, `test_dataset_param_coord`, `test_dataset_s2_coords`,
`test_build_multi_index_first_row_is_zero`, `test_s2_pair_mask_matches_dense` (mirror of
the source's own mask-and-einsum), plus C1/C2.

### tests/test_pce_streaming.py
DELETE: the whole `TestMemoryBudgetConfig` class (five tests, all duplicating
`test_config.py`), `test_single_pass_fit_bytes_formula` (mirror, all 4 params),
`test_loo_two_pass_exactness`, `test_default_budget_keeps_single_pass`,
`test_explicit_batch_size_engages_streaming`.

### tests/test_optimal_transport.py
DELETE: `test_bad_n_bootstrap`, `test_bad_max_iter`, `test_shape_scalar_output`,
`test_multivariate_shape`, `test_trajectory_shape`, `test_multi_output_shape`,
`test_time_series_shape`, `test_row_mismatch`, `test_scalar_dims`, `test_trajectory_dims`,
`test_conf_and_dummy_vars`, `test_time_series_dims`, plus C1/C2/C3.

### tests/test_emulate_batching.py
DELETE: `test_auto_batch_default_matches_explicit` (under the default budget both sides
issue the identical single-shot call), `test_surrogate_template_validates_and_batches`
(pins exact chunk boundaries `[4, 4, 2]`) — but first move its "the kernel never ran on
invalid input" assertion into `test_pce_hdmr_reject_wrong_shaped_x_identically`.
COLLAPSE: `test_pce_batched_matches_single_shot` and `test_hdmr_batched_matches_single_shot`
from 12 cases each to `batch_size in [1, 100]` — `N_NEW` and `10_000` both degenerate
to the single-shot call being compared against.

### tests/test_correlated_agreement.py
DELETE: `test_routes_agree_on_the_reading`.

### tests/test_efast.py
DELETE: `test_returns_efast_samples`, `TestSampling::test_shape`,
`TestSampling::test_invalid_m_raises`, `test_gaussian_inputs`, `test_result_shapes`,
`test_no_s2` (pins `set(vars(result).keys())` to an exact field list),
`test_omega_and_m`, `test_analyze_uses_samples_m` (mirror: recomputes `omega_0` from
the source's formula and re-runs the source's own kernel), `TestFrequencyAssignment::test_d2`,
`TestFrequencyAssignment::test_high_omega`, both `TestFrequencyPlanSharing` tests,
`TestComputeIndices::test_constant_output`, `TestComputeIndices::test_single_frequency`,
`TestToDatasetMultiOutput::test_repr`, `TestToDataset::test_conversion`,
`test_drop_is_refused_even_when_the_sample_is_clean`,
`test_one_bad_value_is_reported_against_its_own_curve`, plus C1.
TRIM: `test_3d_dataset_with_time_coords` down to the `time_coords=` kwarg assertion.

### tests/test_analyze.py
DELETE: `test_repeated_no_bootstrap_calls_identical`,
`test_explicit_prenormalize_false_matches_default`,
`test_default_ci_method_matches_explicit_quantile`, `test_invalid_ci_method_raises_value_error`,
`test_supported_ci_method_is_accepted_without_bootstrap`,
`test_repeated_gaussian_bootstrap_calls_identical`,
`test_prenormalize_gaussian_bootstrap_is_offset_invariant`,
`test_unique_gaussian_bootstrap_matches_expanded_layout`,
`test_unique_bootstrap_prenormalize_matches_expanded_layout`,
`test_gaussian_conf_shapes_match_quantile`,
`test_gaussian_and_quantile_bootstrap_endpoints_differ`.
Rationale: three properties were each cross-producted over {quantile, gaussian} x
{prenormalize, plain}; the endpoint rule and prenormalization are orthogonal to all
three. Keep one case per property.

### tests/test_copula.py
DELETE: `test_canonicalize_correlation_accepts_valid_matrix_unchanged`,
`test_repair_warning_reports_min_eigenvalue_and_max_change`,
`test_material_repair_error_names_the_way_out`, `test_canonicalize_spearman_applies_conversion`,
`test_is_independent_and_identity_helper`, `test_sobol_sample_rejects_correlated_problem`,
`test_morris_sample_rejects_correlated_problem`, `test_efast_sample_rejects_correlated_problem`,
`test_pce_analyze_rejects_correlated_problem`, `test_shapley_pce_backend_rejects_correlated_problem`,
`test_analyzer_guard_message_names_alternatives`,
`test_correlation_tolerant_analyzers_accept_correlated_problem` (a hand-written list of
which methods accept correlation — exactly the drift the registry test prevents),
`test_conditional_plan_chol_full_is_appended_last` (pins `_ConditionalPlan._fields`).
REWRITE (do not delete): `test_sampler_guard_message_names_alternatives` currently
hard-codes the generated list's ordering — assert against `_correlation_tolerant_methods()`
instead. It is the only test of the *sampler* message.

### tests/test_indices.py
DELETE: `test_first_order_jit`, `test_total_order_jit`, `test_second_order_jit` — all
three assert only `isfinite` on arbitrary numbers. `TestKnownValues` stays in full.

### tests/test_imports.py
DELETE: `test_root_exports_foundational_types`,
`test_every_method_namespace_is_reachable_from_the_root`,
`test_every_method_exposes_analyze_and_a_result_type`,
`test_a_design_based_method_exposes_sample_and_a_samples_type`,
`test_prediction_and_shapley_are_result_methods`.
KEEP `test_removed_root_shortcuts_are_absent` (pins names that must NOT exist — the
registry cannot express this) and `test_root_exports_the_support_namespaces`.

### tests/test_problem.py
DELETE: `test_from_dict_accepts_one_or_two_sided_gaussian_truncation`,
`test_direct_constructor_remains_uniform_only`, `test_frozen`,
`test_correlation_defaults_to_none_and_independent`,
`test_from_dict_accepts_latent_correlation`, `test_direct_constructor_accepts_correlation`,
`test_materially_indefinite_correlation_is_rejected_at_construction`,
`test_with_correlation_accepts_spearman_kind`, `test_correlated_problem_is_still_frozen`.

### tests/test_shapley.py
DELETE: `test_build_membership`, `test_default_backend_is_pce`, `test_time_series_shapes`,
`test_to_dataset_scalar`, `test_repr`, `test_sh_sums_to_one`,
`test_correlative_matches_hdmr_sa_plus_sb` (mirror: transcribes `HDMRResult.shapley`
line for line), `test_correlative_provenance_on_result_and_dataset`, `test_hdmr_order_field`,
`test_pce_time_series_to_dataset`, `test_a_non_finite_input_is_caught_too`,
`test_on_invalid_is_not_swallowed_by_backend_kwargs` (reads `inspect.signature`),
plus C1/C2.
TRIM: `test_correlative_explained_variance_is_r2` — drop the mirror equality, keep the
`0 < ev <= 1.05` bound.
KEEP `test_the_policy_is_applied_exactly_once` — double application is invisible in the
indices, so the warning count is the only observable. This is the one warning-count
test in the suite that is irreplaceable.

### tests/test_hsic.py
DELETE: `TestLinearHSIC::test_shapes`, `test_all_positive`,
`TestIshigamiHSIC::test_r2_bounded`, `TestMultiOutput::test_shapes`, `test_shapes_3d`,
`test_fixed_bandwidth`, `test_prenormalize_runs`, `TestToDataset::test_scalar_output`,
`TestToDataset::test_multi_output`, `TestGaussianInputs::test_gaussian_problem`,
`TestSingleParam::test_single_input`, plus C1/C2/C3/C4.
COLLAPSE: `test_quantile_equals_upper_triangle_median` from 11 sizes to `[4, 5, 6, 257]`
(the test is a strong oracle; only the parametrisation is padding). Merge the four
`test_bandwidth_*_raises` functions into one parametrisation over `[0.0, -1.0, nan, inf]`.

### tests/test_sampling.py
DELETE: `test_no_second_order_expanded_count`, `test_sample_verbose_prints_summary`,
`test_mixed_distributions_preserve_sampling_metadata`, `test_n_expanded_matches_step`,
`test_expanded_to_unique_is_consistent`, `test_base_n_stored`, `test_first_order_only`,
`test_single_param_with_duplicates`, `test_problem_preserved`.
TRIM: inside `test_sample_returns_unique_rows`, drop the `_saltelli_step` mirror line;
keep the rest of the test.

### tests/test_pawn.py
DELETE: `test_shape_scalar_output`, `TestPAWNStatistics::test_max_statistic`,
`TestPAWNStatistics::test_mean_statistic`, `test_bootstrap_produces_conf`,
`test_multi_output_shape`, `test_time_series_shape`, `test_scalar_dataset`,
`test_multi_output_dataset`, `test_bootstrap_dataset`, `test_a_bad_x_is_caught_and_named`
(the test immediately below makes the same call and adds the regression claim),
plus C1/C2/C3/C4.
KEEP `test_slice_chunk_size_splits_the_columns` despite the monkeypatch — its docstring
argues correctly that chunking has no other observable effect. Deliberate exception.

### tests/test_morris.py
DELETE: `test_matches_numpy_reference`, `test_radial_matches_numpy_reference` — and
delete the `_numpy_reference` helper with them; it transcribes `_elementary_effects` +
`_stats_from_ee` line for line. Also `test_nonfinite_trajectory_dropped`,
`TestToDataset::test_conf_variables`, `TestToDataset::test_scalar_output`,
`test_y_length_mismatch_raises`, `test_y_wrong_ndim_raises`,
`test_downsample_clears_the_earlier_block_loss`, `test_analysis_detects_gaussian_input`,
`test_radial_gaussian_finite`, `TestMultiOutput::test_shapes`,
`test_verbose_reports_duplicates`, plus C1.

### tests/test_borgonovo.py
DELETE: `test_multi_output_dataset`, `test_rejects_bad_grid_size`, `test_scalar_dataset`,
`test_rejects_mismatched_rows`, `test_rejects_4d_y` (KEEP `test_rejects_0d_y` — the 0-d
case is unique), `test_constant_column_among_varying`, `test_shape_scalar_output`,
`test_bootstrap_produces_conf`, `test_multi_output_shape`,
`test_continuous_fixture_never_engages_floor`, `test_message_names_the_floored_column`,
`test_rejects_bad_conf_level`, `test_saturates_at_48`, plus C1/C2/C3.
TRIM: `test_auto_floor_still_aliases_under_a_tiny_bandwidth_factor` — drop the asserts
that quote the source's own floor formula and its rendered number; keep the closing
block (raising `grid_size` fixes the run). Same for
`test_aliased_floor_raises_and_names_the_knobs`: keep the `pytest.raises`, drop the four
literal substring asserts derived from the floor formula.
COLLAPSE: `test_explicit_bandwidth_that_cannot_apply_is_accepted` from 4 cases to
`(100, 0.1)` and `(50, 0.3)`.

### tests/test_to_morris.py
DELETE: `test_increments_equal_delta_times_ee` (computes `ee = d/delta` then asserts
`ee*delta == d` — float arithmetic asserting itself), `test_matches_numpy_reference`
(and its `_numpy_reference` helper), `test_ba_rows_duplicate_additive_effects_only`
(never calls `to_morris`), `test_all_parameters_detected`, `test_downsample_then_analyze`,
`test_verbose_summary`.

### tests/test_output_shapes.py
DELETE: `test_canonical_shapes_pass`, `test_two_dimensional_output_is_always_n_k`.

### tests/test_shapes.py
DELETE: `test_no_bootstrap_conf_is_none`. The rest of this file is a clean,
non-overlapping shape matrix — leave it alone.

### tests/test_sobol_gradients.py
COLLAPSE ONLY: `test_unit_cube_dedup_gives_the_same_design` — the three continuous
marginals (uniform/gaussian/truncated) are all strictly monotone and exercise one
injectivity path; keep one of them plus the `categorical` case, and keep the `scramble`
and `calc_second_order` axes. No deletions.

### tests/test_invalid_policy.py
DELETE: `test_clean_report_records_that_the_check_ran`, `test_accepts_the_three_policies`,
`test_repr_abbreviates_a_long_index_list`, `test_default_names_are_still_x_and_y`,
`test_indices_refer_to_the_original_numbering`.

### tests/test_kucherenko.py
DELETE: `test_nonfinite_outputs_drop_base_points_with_warning`, `test_repr`,
`test_categorical_problem_raises`, `test_design_uses_the_factor_on_the_plan`,
`test_to_dataset_schema`.

### tests/test_entry_preamble.py
DELETE: all four `TestTheGatesComeFromTheRegistry` list tests
(`test_the_correlation_list_names_every_tolerant_method`,
`test_the_correlation_list_names_no_refusing_method`,
`test_the_categorical_list_names_every_tolerant_method`,
`test_the_categorical_list_names_no_refusing_method`). These recompute the source's own
filter (`validation.py:154-158`) against its own output, and the "no refusing method"
variant needs a hand-written `!= "shapley"` carve-out to stay true — a mirror with an
epicycle. KEEP `test_a_refusing_method_quotes_the_generated_list` and
`test_the_shapley_pce_message_quotes_it_too`: those assert a real message against a real
refusal.
ALSO DELETE: `test_inputs_returns_the_matrix_a_given_data_method_was_handed`
(`assert ctx.inputs is ctx.X` on a two-line property),
`test_inputs_refuses_when_the_method_was_given_no_matrix` (unreachable branch — keep the
branch, drop the test), `test_morris_names_its_own_three_measures`,
`test_every_bootstrapping_method_now_checks_it` (hard-codes five method names the
registry already declares, and only checks that `conf_level` appears in a signature —
never that it is validated), `test_the_wording_says_nan_for_a_variance_ratio`,
`test_the_wording_says_zero_for_a_distribution_comparison`.
COLLAPSE: `TestConfLevelIsValidatedEverywhere` from `[-0.2, 0.0, 1.0, 1.5]` to one bad
value per method — `test_in_open_interval_rejects_both_endpoints` already covers all
four against the predicate. `TestChunkSizeIsValidatedOnEveryPath` from `[0, -5]` to one
value; the three *paths* are the point and all three stay.

### tests/test_registry.py
DELETE: `test_the_analyze_and_result_entries_are_the_exported_ones` (13 cases of
`spec.analyze is namespace.analyze`, where `__init__.py` writes `analyze=analyze` two
lines below the import — a name compared with itself),
`test_a_design_based_method_exposes_its_sampler`,
`test_the_walk_finds_the_packages_it_claims_to_check`,
`test_the_returned_mapping_cannot_be_edited`,
`test_the_registry_matches_the_public_namespace_list` (its hand-maintained
`-= {"benchmarks", "config", "sampling"}` exclusion must be edited whenever a non-method
package is added).
KEEP both capability probes and `test_every_method_package_on_disk_is_registered` —
those are the file's justification.

### tests/test_docs_matrix.py
DELETE: `test_the_parser_finds_the_table_it_claims_to_check`,
`test_the_prose_counts_the_design_based_methods_correctly`,
`test_the_footnote_on_the_bootstrap_column_counts_correctly` (requires a literal English
phrase and keeps a private number-speller in the test to match it),
`test_the_rows_are_in_alphabetical_order` (a cosmetic convention).
Also delete the now-unused `_spelled()` helper.
NOTE: `EXPECTED_COLUMNS` declares a `"Reports"` column that no test reads — so that
column can currently say anything. Either assert it or drop it from the constant.

### tests/test_result_schema.py
DELETE: `test_every_registered_result_declares_a_schema` (reads `cls._schema` and
asserts a declaration is a declaration), `test_ot_trajectory_shape_comes_from_the_mode`,
`test_ci_records_the_endpoint_rule_that_ran`, `test_sobol_records_its_ci`,
`test_hdmr_term_axis_is_not_labelled_param`.
TRIM: `test_ciinfo_repr_hides_the_stored_draws` — drop the three exact-repr-substring
asserts, keep the last line (the actual contract).

### tests/test_xarray.py
DELETE: `test_s2_confidence`, `TestHDMRResultToDataset::test_2d`,
`test_with_select_and_rmse`, `test_field_stored`, `test_default_none`, `test_s2_dims`,
`test_confidence_intervals`, `test_from_dict_output_names`.
COLLAPSE: `test_1d`, `test_2d_default_output_names`, `test_3d` into one test asserting
only the coordinate *values* (`y0`/`y1`, integer time) — the dims are snapshot-covered.

### tests/test_config.py
DELETE: `test_internal_getter_always_returns_bytes`,
`test_guard_does_not_fire_with_explicit_unit`.
COLLAPSE: `test_each_unit_resolves_to_its_exact_byte_count` from 10 cases to the two
carrying the behavioural claim (`kb`=1024, `b`=1) plus one `*ib` synonym;
`test_unit_ignores_case_and_whitespace` from 6 spellings to 1;
`test_zero_and_negative_rejected` from 5 values to 2;
`test_unknown_unit_raises` from 5 spellings to 1, and drop the `"mib" in message` assert.
KEEP `test_default_budget_is_unchanged_in_bytes` (writes `536870912` longhand precisely
so it cannot follow the source) and `TestBytesShapedGuard::test_guard_fires_without_unit`.

### tests/test_warning_category.py
DELETE: `test_jaxgsa_warning_is_a_user_warning_subclass`,
`test_the_walk_finds_the_warnings_it_claims_to_check` (a floor of 30 that guards the
test module's own AST helper and needs revisiting as the package grows).
KEEP `test_every_warning_passes_a_category` — it earns the whole file.

### tests/test_cleaning.py
DELETE: `test_drop_nonfinite_rows`, `test_inf_values_dropped`,
`test_first_order_zero_variance`, `test_total_order_zero_variance`,
`test_second_order_zero_variance` (the last three call private sobol kernels to assert
NaN-not-inf, which the public-surface tests already assert), plus C1.

### tests/test_save_load.py
DELETE: `test_metadata_records_jaxgsa_version`,
`test_sobol_and_morris_share_metadata_schema` (mirrors the dict literal in
`_problem_to_meta`), `test_morris_identity_mapping_skips_index_array`,
`test_morris_npz_round_trip_carries_identity_correlation`,
`test_problem_meta_round_trip_preserves_none_correlation`,
`test_duplicate_rows_store_index_array` (keep `test_identity_mapping_skips_index_array`
— one of the pair is enough).

### tests/test_baseline_check.py
DELETE: `test_the_default_still_treats_a_schema_change_as_a_failure` (identical setup and
assertion to `test_a_schema_change_alone_exits_two`),
`test_diffs_is_falsy_only_when_both_are_empty`, `test_a_gained_method_is_a_schema_change`.

## Production changes

**P1. `_HDMRStaticData` is built positionally while every consumer reads it by name.**
`tests/test_hdmr_streaming.py::TestHDMRStaticDataLayout::test_field_order_is_unchanged`
exists purely to guard that fragility by pinning `_fields`. Fix the source — construct
`_HDMRStaticData` with keyword arguments — then delete the test. Deleting the test
without the fix moves risk rather than removing it.

**P2. `dgsm/_poincare.py:34` `poincare_constant(spec, *, grid: int = 512)`.** Nothing in
`src/` or `tests/` ever passes `grid`; `axis_constants` calls it positionally, and the
two tests that vary the mesh call `_truncnorm_poincare` directly. Delete the parameter
and inline `512`. It is a promise of configurability no caller takes up.

**P3. `tests/_result_fixtures.py` has a fictional second consumer.** Its docstring says
"the generator that writes the snapshot imports the same builders". No such generator
exists — the module has exactly one importer, and `tests/data/result_dataset_schema.json`
therefore has no regeneration path at all. Write `scripts/dump_result_schema.py`
(mirroring the existing `scripts/baseline_dump.py`) so the snapshot can be regenerated,
which also makes the module's split real. Do NOT simply fix the docstring — the missing
generator is the actual defect, and the snapshot gate is unusable without it.

**P4. `_core/entry.py:319-322`** carries `# pragma: no cover - guarded by
tests/test_registry.py`, but no test reaches or asserts that branch. Correct the comment
to say it is defensive; do not claim coverage that does not exist.

**P5. `MethodSpec.analyze`, `.result`, `.bootstrap` are never read in `src/`.**
Verified: the only `spec.*` reads in production are `name`, `correlation`, `categorical`,
`invalid_unit` and `is_design_based`. Do NOT delete these fields — the registry exists to
support generic dispatch, and `result`/`bootstrap` are what make the documentation table
answerable. Delete only the tautological test that asserts `spec.analyze is
namespace.analyze`. Recorded here so the next reader knows the fields are declarative.

**P6 (flag only, no change).** `jaxgsa.config` has no way to return the memory budget to
"unset" — `tests/test_config.py` writes the module global `_memory_budget_bytes` directly
to restore it, with a comment saying the internal setter cannot express `None`. That is a
missing public API (`config.reset_memory_budget()`), not an extra one. Left for the user
to decide.

## Verification required after every deletion batch

1. `uv run pytest -q` — must stay green.
2. `uv run python scripts/baseline_check.py --quiet` — **must report zero moved values.**
   Deleting tests cannot change a number; if a value moves, something real broke.
3. `uv run ruff format` and `uv run ruff check`.
4. `uv run ty check src/jaxgsa` (and any edited script).
5. No `# noqa` or `# type: ignore` added.
