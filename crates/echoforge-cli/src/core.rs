use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Ok,
    Warn,
    Fail,
}

impl Health {
    pub fn label(self) -> &'static str {
        match self {
            Health::Ok => "ok",
            Health::Warn => "warn",
            Health::Fail => "fail",
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Health::Ok => "OK",
            Health::Warn => "WARN",
            Health::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusCheck {
    pub name: String,
    pub health: Health,
    pub detail: String,
}

impl StatusCheck {
    pub fn new(name: impl Into<String>, health: Health, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            health,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusSummary {
    pub title: String,
    pub checks: Vec<StatusCheck>,
}

impl StatusSummary {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            checks: Vec::new(),
        }
    }

    pub fn push(&mut self, check: StatusCheck) {
        self.checks.push(check);
    }

    pub fn overall(&self) -> Health {
        if self.checks.iter().any(|check| check.health == Health::Fail) {
            Health::Fail
        } else if self.checks.iter().any(|check| check.health == Health::Warn) {
            Health::Warn
        } else {
            Health::Ok
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{}", self.title);
        let _ = writeln!(out, "overall: {}", self.overall().label());

        for check in &self.checks {
            let _ = writeln!(
                out,
                "- {}: {} ({})",
                check.name,
                check.health.label(),
                check.detail
            );
        }

        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSummary {
    pub path: std::path::PathBuf,
    pub artifact: Option<String>,
    pub validation_tier: Option<String>,
    pub status: StatusSummary,
}

impl ManifestSummary {
    pub fn render(&self) -> String {
        let mut out = self.status.render();
        let _ = writeln!(out, "manifest: {}", self.path.display());
        if let Some(artifact) = &self.artifact {
            let _ = writeln!(out, "artifact: {}", artifact);
        }
        if let Some(validation_tier) = &self.validation_tier {
            let _ = writeln!(out, "validation_tier: {}", validation_tier);
        }
        out
    }
}
