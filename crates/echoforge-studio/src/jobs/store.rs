use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::results::empty_results;
use super::types::{JobComposeRequest, JobSummary};

#[derive(Debug, Clone)]
pub struct JobStore {
    jobs: Arc<Mutex<Vec<JobSummary>>>,
}

impl JobStore {
    pub fn seeded(repo_root: &std::path::Path) -> Self {
        let request = JobComposeRequest::default();
        let job = seeded_job(repo_root, request);
        Self {
            jobs: Arc::new(Mutex::new(vec![job])),
        }
    }

    pub fn create(&self, request: JobComposeRequest) -> JobSummary {
        let job = pending_job(request);
        self.jobs.lock().expect("job store").push(job.clone());
        job
    }

    pub fn list(&self) -> Vec<JobSummary> {
        self.jobs.lock().expect("job store").clone()
    }

    pub fn get(&self, job_id: &str) -> Option<JobSummary> {
        self.jobs
            .lock()
            .expect("job store")
            .iter()
            .find(|job| job.job_id == job_id)
            .cloned()
    }

    pub fn cancel(&self, job_id: &str) -> Option<JobSummary> {
        self.update(job_id, |job| {
            if job.status != "completed" {
                job.status = "cancelled".to_string();
                job.message = "job cancelled before ML processing completed".to_string();
                job.progress_percent = job.progress_percent.min(95);
            }
        })
    }

    pub fn update<F>(&self, job_id: &str, mut update: F) -> Option<JobSummary>
    where
        F: FnMut(&mut JobSummary),
    {
        let mut jobs = self.jobs.lock().expect("job store");
        let job = jobs.iter_mut().find(|job| job.job_id == job_id)?;
        update(job);
        Some(job.clone())
    }
}

fn pending_job(request: JobComposeRequest) -> JobSummary {
    let job_hash = job_hash(&request);
    let job_id = format!("job-ml-{}-{job_hash}", request.selection);
    JobSummary {
        job_id,
        created_utc: now_utc_compact(),
        status: "queued".to_string(),
        progress_percent: 0,
        message: "ML processing job queued".to_string(),
        request: request.clone(),
        record_count: 0,
        output_dir: request.out_root.clone(),
        artifacts: Vec::new(),
        results: empty_results(&request, "queued", "ML processing job queued"),
    }
}

fn job_hash(request: &JobComposeRequest) -> String {
    let mut hasher = DefaultHasher::new();
    request.job_type.hash(&mut hasher);
    request.selection.hash(&mut hasher);
    request.pipeline_id.hash(&mut hasher);
    request.suite_id.hash(&mut hasher);
    request.seed.hash(&mut hasher);
    request.smoke.hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

fn seeded_job(repo_root: &std::path::Path, request: JobComposeRequest) -> JobSummary {
    let output_dir = repo_root
        .join(request.out_root.clone())
        .display()
        .to_string();
    let results = empty_results(&request, "completed", "seeded ML job metadata ready");
    JobSummary {
        job_id: format!("job-ml-{}-seeded", request.selection),
        created_utc: now_utc_compact(),
        status: "completed".to_string(),
        progress_percent: 100,
        message: "seeded ML job metadata ready".to_string(),
        request,
        record_count: 0,
        output_dir,
        artifacts: Vec::new(),
        results,
    }
}

fn now_utc_compact() -> String {
    let secs = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    };
    format!("{secs}Z")
}
