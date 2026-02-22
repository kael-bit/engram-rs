use axum::http::StatusCode;
use axum::Json;

#[derive(Debug, thiserror::Error)]
pub enum EngramError {
    #[error("content must not be empty")]
    EmptyContent,

    #[error("query must not be empty")]
    EmptyQuery,

    #[error("content exceeds maximum length")]
    ContentTooLong,

    #[error("invalid layer: {0} (expected 1, 2, or 3)")]
    InvalidLayer(u8),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("not found")]
    NotFound,

    #[error("unauthorized")]
    Unauthorized,

    #[error("AI not configured (set ENGRAM_LLM_URL)")]
    AiNotConfigured,

    #[error("AI backend error: {0}")]
    AiBackend(String),

    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("internal error: {0}")]
    Internal(String),
}

impl EngramError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::Database(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::AiNotConfigured => StatusCode::SERVICE_UNAVAILABLE,
            Self::AiBackend(_) => StatusCode::BAD_GATEWAY,
            _ => StatusCode::BAD_REQUEST,
        }
    }
}

impl axum::response::IntoResponse for EngramError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status_code();
        let body = Json(serde_json::json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}
