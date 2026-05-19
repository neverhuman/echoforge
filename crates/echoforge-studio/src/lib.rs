use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::get;
use axum::Json;
use axum::Router;
use echoforge_validate::{run_validate, ValidateArgs, ValidateReport};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::net::TcpListener;
use tower_http::services::{ServeDir, ServeFile};

pub const DEFAULT_SERVICE_NAME: &str = "echoforge-studio";
pub const DEFAULT_PUBLIC_BASE_URL: &str = "/";
pub const DEFAULT_CATALOG_PATH: &str = "contracts/schema_catalog.json";
pub const DEFAULT_BUNDLE_PATH: &str = "tests/science/fixtures/bundles/v1_pass";
pub const DEFAULT_WEB_DIST: &str = "apps/web/dist";
pub const DEFAULT_TARGET_TIER: &str = "v1";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogEntry {
    pub name: String,
    pub schema_file: String,
    pub rust_type: String,
    pub python_type: String,
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    schemas: Vec<CatalogEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthResponse {
    pub service: String,
    pub status: String,
    pub mode: String,
    pub public_base_url: String,
    pub catalog_source: String,
    pub schema_count: usize,
    pub bundle_path: String,
    pub validation_status: String,
    pub validation_tier: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogResponse {
    pub service: String,
    pub status: String,
    pub public_base_url: String,
    pub catalog_source: String,
    pub schema_count: usize,
    pub schemas: Vec<CatalogEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContractsResponse {
    pub service: String,
    pub status: String,
    pub public_base_url: String,
    pub catalog_source: String,
    pub bundle_path: String,
    pub health: HealthResponse,
    pub catalog: CatalogResponse,
    pub validation: ValidateReport,
}

#[derive(Debug, Clone)]
pub struct StudioConfig {
    pub host: String,
    pub port: u16,
    pub service_name: String,
    pub public_base_url: String,
    pub catalog_path: PathBuf,
    pub bundle_path: PathBuf,
    pub web_dist: PathBuf,
}

impl StudioConfig {
    pub fn from_env() -> Result<Self, StudioError> {
        let host = match std::env::var("HOST") {
            Ok(v) => v,
            Err(_) => "127.0.0.1".to_string(),
        };
        let port = match std::env::var("PORT") {
            Ok(v) => v,
            Err(_) => "8080".to_string(),
        }
        .parse::<u16>()
        .map_err(|err| StudioError::Config(format!("invalid PORT: {err}")))?;
        let service_name = match std::env::var("ECHOFORGE_SERVICE_NAME") {
            Ok(v) => v,
            Err(_) => DEFAULT_SERVICE_NAME.to_string(),
        };
        let public_base_url = match std::env::var("ECHOFORGE_PUBLIC_BASE_URL") {
            Ok(v) => v,
            Err(_) => DEFAULT_PUBLIC_BASE_URL.to_string(),
        };
        let catalog_path = PathBuf::from(match std::env::var("ECHOFORGE_CATALOG_PATH") {
            Ok(v) => v,
            Err(_) => DEFAULT_CATALOG_PATH.to_string(),
        });
        let bundle_path = PathBuf::from(match std::env::var("ECHOFORGE_BUNDLE_PATH") {
            Ok(v) => v,
            Err(_) => DEFAULT_BUNDLE_PATH.to_string(),
        });
        let web_dist = PathBuf::from(match std::env::var("ECHOFORGE_WEB_DIST") {
            Ok(v) => v,
            Err(_) => DEFAULT_WEB_DIST.to_string(),
        });

        Ok(Self {
            host,
            port,
            service_name,
            public_base_url,
            catalog_path,
            bundle_path,
            web_dist,
        })
    }
}

#[derive(Debug, Error)]
pub enum StudioError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("validation error: {0}")]
    Validation(#[from] echoforge_validate::ValidateError),
    #[error("config error: {0}")]
    Config(String),
    #[error("missing asset: {0}")]
    MissingAsset(String),
}

#[derive(Debug, Clone)]
pub struct StudioState {
    pub service_name: String,
    pub public_base_url: String,
    pub catalog_source: String,
    pub bundle_path: String,
    pub mode: String,
    pub catalog: CatalogResponse,
    pub validation: ValidateReport,
}

impl StudioState {
    pub fn load(config: &StudioConfig) -> Result<Self, StudioError> {
        let index = config.web_dist.join("index.html");
        if !index.exists() {
            return Err(StudioError::MissingAsset(format!(
                "missing Vite build output: {}",
                index.display()
            )));
        }

        let catalog_file: CatalogFile = load_json(&config.catalog_path)?;
        let catalog_entries = catalog_file.schemas;
        let catalog_source = config.catalog_path.display().to_string();
        let validation = load_validation_report(&config.bundle_path)?;
        let catalog = CatalogResponse {
            service: config.service_name.clone(),
            status: "ready".to_string(),
            public_base_url: config.public_base_url.clone(),
            catalog_source: catalog_source.clone(),
            schema_count: catalog_entries.len(),
            schemas: catalog_entries,
        };

        Ok(Self {
            service_name: config.service_name.clone(),
            public_base_url: config.public_base_url.clone(),
            catalog_source,
            bundle_path: config.bundle_path.display().to_string(),
            mode: "rust-studio".to_string(),
            catalog,
            validation,
        })
    }

    fn health(&self) -> HealthResponse {
        HealthResponse {
            service: self.service_name.clone(),
            status: "ok".to_string(),
            mode: self.mode.clone(),
            public_base_url: self.public_base_url.clone(),
            catalog_source: self.catalog_source.clone(),
            schema_count: self.catalog.schema_count,
            bundle_path: self.bundle_path.clone(),
            validation_status: self.validation.overall_status.clone(),
            validation_tier: self.validation.tier.as_str().to_string(),
        }
    }

    fn contracts(&self) -> ContractsResponse {
        ContractsResponse {
            service: self.service_name.clone(),
            status: "ready".to_string(),
            public_base_url: self.public_base_url.clone(),
            catalog_source: self.catalog_source.clone(),
            bundle_path: self.bundle_path.clone(),
            health: self.health(),
            catalog: self.catalog.clone(),
            validation: self.validation.clone(),
        }
    }
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, StudioError> {
    let raw = fs::read_to_string(path)?;
    let value = serde_json::from_str(&raw)?;
    Ok(value)
}

fn load_validation_report(bundle_path: &Path) -> Result<ValidateReport, StudioError> {
    let tmp_file = NamedTempFile::new()?;
    let report_path = tmp_file.path().to_path_buf();
    let args = ValidateArgs {
        bundle: bundle_path.to_path_buf(),
        primitive: Some("auto".to_string()),
        target_tier: DEFAULT_TARGET_TIER.to_string(),
        write_report: Some(report_path.clone()),
        strict: false,
    };
    let _ = run_validate(args)?;
    let raw = fs::read_to_string(&report_path)?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn build_router(state: Arc<StudioState>, web_dist: PathBuf) -> Router {
    let health_state = state.clone();
    let catalog_state = state.clone();
    let validation_state = state.clone();
    let contracts_state = state;
    let index = web_dist.join("index.html");
    let static_files = ServeDir::new(&web_dist).not_found_service(ServeFile::new(index));

    Router::new()
        .route(
            "/healthz",
            get(move || {
                let state = health_state.clone();
                async move { Json(state.health()) }
            }),
        )
        .route(
            "/api/catalog",
            get(move || {
                let state = catalog_state.clone();
                async move { Json(state.catalog.clone()) }
            }),
        )
        .route(
            "/api/validation/latest",
            get(move || {
                let state = validation_state.clone();
                async move { Json(state.validation.clone()) }
            }),
        )
        .route(
            "/api/contracts",
            get(move || {
                let state = contracts_state.clone();
                async move { Json(state.contracts()) }
            }),
        )
        .fallback_service(static_files)
}

#[tracing::instrument(name = "studio.serve_from_env")]
pub async fn serve_from_env() -> Result<(), StudioError> {
    let config = StudioConfig::from_env()?;
    serve(config).await
}

#[tracing::instrument(name = "studio.serve", fields(host = %config.host, port = %config.port))]
pub async fn serve(config: StudioConfig) -> Result<(), StudioError> {
    let state = Arc::new(StudioState::load(&config)?);
    let router = build_router(state, config.web_dist.clone());
    let listener = TcpListener::bind((config.host.as_str(), config.port)).await?;
    axum::serve(listener, router.into_make_service()).await?;
    Ok(())
}

impl IntoResponse for StudioError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::INTERNAL_SERVER_ERROR;
        let body = Json(serde_json::json!({
            "service": DEFAULT_SERVICE_NAME,
            "status": "error",
            "message": self.to_string(),
        }));
        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use std::fs;
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn test_config(web_dist: PathBuf) -> StudioConfig {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root");
        StudioConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            service_name: DEFAULT_SERVICE_NAME.to_string(),
            public_base_url: DEFAULT_PUBLIC_BASE_URL.to_string(),
            catalog_path: workspace_root.join("contracts/schema_catalog.json"),
            bundle_path: workspace_root.join("tests/science/fixtures/bundles/v1_pass"),
            web_dist,
        }
    }

    #[tokio::test]
    async fn router_serves_live_contracts() {
        let dist = tempdir().expect("tempdir");
        fs::write(
            dist.path().join("index.html"),
            "<!doctype html><div id=\"app\"></div>",
        )
        .expect("index");
        let config = test_config(dist.path().to_path_buf());
        let state = Arc::new(StudioState::load(&config).expect("state"));
        let app = build_router(state, config.web_dist.clone());

        let health = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(health.status(), axum::http::StatusCode::OK);
        let health_body = axum::body::to_bytes(health.into_body(), usize::MAX)
            .await
            .expect("health body");
        let health_json: HealthResponse =
            serde_json::from_slice(&health_body).expect("health json");
        assert_eq!(health_json.service, DEFAULT_SERVICE_NAME);
        assert_eq!(health_json.validation_status, "pass");
        assert_eq!(health_json.schema_count, 12);

        let validation = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/validation/latest")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(validation.status(), axum::http::StatusCode::OK);
        let validation_body = axum::body::to_bytes(validation.into_body(), usize::MAX)
            .await
            .expect("validation body");
        let validation_json: ValidateReport =
            serde_json::from_slice(&validation_body).expect("validation json");
        assert_eq!(validation_json.overall_status, "pass");
        assert_eq!(validation_json.tier.as_str(), "V1");

        let contracts = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/contracts")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(contracts.status(), axum::http::StatusCode::OK);
    }
}
