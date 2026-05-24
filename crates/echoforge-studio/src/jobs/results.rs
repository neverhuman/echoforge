use echoforge_dataset::{
    run_pipeline, run_suite, PipelineRunRequest, PipelineRunResult, PipelineSuiteResult,
};

use super::types::{
    CalibrationBin, JobArtifact, JobComposeRequest, JobExecutionPayload, JobResults, MetricSlice,
    PipelineSummary, ValidationGate,
};

pub(super) fn execute_job(
    request: &JobComposeRequest,
    repo_root: &std::path::Path,
) -> Result<JobExecutionPayload, String> {
    if request.selection == "suite" {
        let result = run_suite(
            &request.suite_id,
            Some(repo_root.to_path_buf()),
            std::path::PathBuf::from(&request.data_root),
            std::path::PathBuf::from(&request.out_root),
            request.workers_per_pipeline,
            request.max_concurrent,
            request.seed,
            request.smoke,
            request.validation_tier.clone(),
        )
        .map_err(|err| err.to_string())?;
        Ok(payload_from_suite(request, result))
    } else {
        let result = run_pipeline(PipelineRunRequest {
            pipeline_id: request.pipeline_id.clone(),
            data_root: std::path::PathBuf::from(&request.data_root),
            out_root: std::path::PathBuf::from(&request.out_root),
            workers: request.workers_per_pipeline,
            seed: request.seed,
            smoke: request.smoke,
            validation_tier: request.validation_tier.clone(),
            repo_root: Some(repo_root.to_path_buf()),
        })
        .map_err(|err| err.to_string())?;
        Ok(payload_from_pipeline(request, result))
    }
}

pub(super) fn empty_results(
    request: &JobComposeRequest,
    status: &str,
    message: &str,
) -> JobResults {
    JobResults {
        primary_pipeline_id: request.pipeline_id.clone(),
        suite_id: if request.selection == "suite" {
            Some(request.suite_id.clone())
        } else {
            None
        },
        auc_table_path: String::new(),
        roc_points: Vec::new(),
        pr_auc: 0.0,
        phase_auc: Vec::new(),
        sensor_holdout_auc: Vec::new(),
        class_holdout_auc: Vec::new(),
        hard_negative_breakdown: Vec::new(),
        leakage_gates: vec![ValidationGate {
            gate: status.to_string(),
            status: "failed".to_string(),
            detail: message.to_string(),
        }],
        calibration_bins: Vec::new(),
        export_readiness: status.to_string(),
        pipeline_summaries: Vec::new(),
    }
}

fn payload_from_pipeline(
    request: &JobComposeRequest,
    result: PipelineRunResult,
) -> JobExecutionPayload {
    let primary_summary = PipelineSummary {
        pipeline_id: result.pipeline_id.clone(),
        status: result.status.clone(),
        output_dir: result.output_dir.clone(),
        roc_auc: metric_value(&result.metrics, &["overall", "roc_auc"]).unwrap_or(0.5),
        pr_auc: metric_value(&result.metrics, &["overall", "pr_auc"]).unwrap_or(result.pr_auc()),
        missing_input_kind: result.missing_input_kind.clone(),
    };
    let output_dir = result.output_dir.clone();
    let result_status = result.status.clone();
    let record_count = metric_value(&result.metrics, &["overall", "count"])
        .unwrap_or(0.0)
        .round() as usize;
    let leakage_gates = result
        .gates
        .iter()
        .cloned()
        .map(|gate| ValidationGate {
            gate: gate.gate,
            status: gate.status,
            detail: gate.detail,
        })
        .collect();
    let results = JobResults {
        primary_pipeline_id: result.pipeline_id.clone(),
        suite_id: None,
        auc_table_path: format!("{output_dir}/metrics.json"),
        roc_points: result.roc_points_from_metrics(),
        pr_auc: primary_summary.pr_auc,
        phase_auc: result.metric_rows("phase"),
        sensor_holdout_auc: result.metric_rows("sensor_holdout"),
        class_holdout_auc: result.metric_rows("class_holdout"),
        hard_negative_breakdown: result.metric_rows("hard_negative_breakdown"),
        leakage_gates,
        calibration_bins: result.calibration_bins(),
        export_readiness: if result_status == "completed" {
            "ready".to_string()
        } else {
            result_status
        },
        pipeline_summaries: vec![primary_summary],
    };
    let artifacts = result
        .artifacts
        .into_iter()
        .map(|artifact| JobArtifact {
            id: artifact.id,
            kind: artifact.kind,
            path: artifact.path,
            ready: artifact.ready,
            incomplete: artifact.incomplete,
        })
        .collect();
    JobExecutionPayload {
        message: format!(
            "{} job completed via pipeline {}",
            request.selection, request.pipeline_id
        ),
        record_count,
        output_dir,
        artifacts,
        results,
    }
}

fn payload_from_suite(
    request: &JobComposeRequest,
    result: PipelineSuiteResult,
) -> JobExecutionPayload {
    let suite_id = result.suite.clone();
    let primary = result
        .results
        .first()
        .cloned()
        .unwrap_or_else(|| PipelineRunResult {
            pipeline_id: request.suite_id.clone(),
            run_id: String::new(),
            status: "failed".to_string(),
            message: "suite produced no results".to_string(),
            output_dir: request.out_root.clone(),
            artifacts: Vec::new(),
            metrics: serde_json::json!({}),
            gates: Vec::new(),
            notes: Vec::new(),
            missing_input_kind: None,
            worker_count: request.workers_per_pipeline,
            error_code: None,
            details: None,
        });
    let primary_status = primary.status.clone();
    let pipeline_summaries = result
        .results
        .iter()
        .map(|item| PipelineSummary {
            pipeline_id: item.pipeline_id.clone(),
            status: item.status.clone(),
            output_dir: item.output_dir.clone(),
            roc_auc: metric_value(&item.metrics, &["overall", "roc_auc"]).unwrap_or(0.5),
            pr_auc: metric_value(&item.metrics, &["overall", "pr_auc"]).unwrap_or(0.5),
            missing_input_kind: item.missing_input_kind.clone(),
        })
        .collect::<Vec<_>>();
    let artifacts = result
        .results
        .iter()
        .flat_map(|item| {
            item.artifacts.iter().cloned().map(|artifact| JobArtifact {
                id: artifact.id,
                kind: artifact.kind,
                path: artifact.path,
                ready: artifact.ready,
                incomplete: artifact.incomplete,
            })
        })
        .collect();
    let output_dir = primary.output_dir.clone();
    let record_count = pipeline_summaries.len();
    let leakage_gates = primary
        .gates
        .iter()
        .cloned()
        .map(|gate| ValidationGate {
            gate: gate.gate,
            status: gate.status,
            detail: gate.detail,
        })
        .collect();
    let results = JobResults {
        primary_pipeline_id: primary.pipeline_id.clone(),
        suite_id: Some(suite_id),
        auc_table_path: format!("{output_dir}/metrics.json"),
        roc_points: primary.roc_points_from_metrics(),
        pr_auc: metric_value(&primary.metrics, &["overall", "pr_auc"]).unwrap_or(0.5),
        phase_auc: primary.metric_rows("phase"),
        sensor_holdout_auc: primary.metric_rows("sensor_holdout"),
        class_holdout_auc: primary.metric_rows("class_holdout"),
        hard_negative_breakdown: primary.metric_rows("hard_negative_breakdown"),
        leakage_gates,
        calibration_bins: primary.calibration_bins(),
        export_readiness: if primary_status == "completed" {
            "ready".to_string()
        } else {
            primary_status
        },
        pipeline_summaries,
    };
    JobExecutionPayload {
        message: format!("suite {} completed", request.suite_id),
        record_count,
        output_dir,
        artifacts,
        results,
    }
}

fn metric_value(metrics: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = metrics;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

trait PipelineResultExt {
    fn pr_auc(&self) -> f64;
    fn roc_points_from_metrics(&self) -> Vec<[f64; 2]>;
    fn metric_rows(&self, key: &str) -> Vec<MetricSlice>;
    fn calibration_bins(&self) -> Vec<CalibrationBin>;
}

impl PipelineResultExt for PipelineRunResult {
    fn pr_auc(&self) -> f64 {
        metric_value(&self.metrics, &["overall", "pr_auc"]).unwrap_or(0.0)
    }

    fn roc_points_from_metrics(&self) -> Vec<[f64; 2]> {
        match self
            .metrics
            .get("overall")
            .and_then(|overall| overall.get("roc_auc"))
        {
            Some(value) => vec![[0.0, 0.0], [0.5, value.as_f64().unwrap_or(0.5)], [1.0, 1.0]],
            None => Vec::new(),
        }
    }

    fn metric_rows(&self, key: &str) -> Vec<MetricSlice> {
        match self.metrics.get(key).and_then(|value| value.as_array()) {
            Some(rows) => rows
                .iter()
                .map(|row| MetricSlice {
                    label: row
                        .get("phase")
                        .or_else(|| row.get("sensor_id"))
                        .or_else(|| row.get("target_family"))
                        .or_else(|| row.get("hard_negative_family"))
                        .or_else(|| row.get("label"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    auc: row
                        .get("roc_auc")
                        .or_else(|| row.get("auc"))
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.5),
                })
                .collect(),
            None => Vec::new(),
        }
    }

    fn calibration_bins(&self) -> Vec<CalibrationBin> {
        match self
            .metrics
            .get("calibration_report")
            .and_then(|value| value.get("bins"))
            .and_then(|value| value.as_array())
        {
            Some(bins) => bins
                .iter()
                .enumerate()
                .map(|(index, bin)| CalibrationBin {
                    bin: index as u32,
                    lower: bin
                        .get("lower")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                    upper: bin
                        .get("upper")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                    count: bin
                        .get("count")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                    mean_score: bin
                        .get("mean_score")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                    positive_rate: bin
                        .get("positive_rate")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0),
                })
                .collect(),
            None => Vec::new(),
        }
    }
}
