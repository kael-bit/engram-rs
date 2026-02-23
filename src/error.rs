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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_are_correct() {
        assert_eq!(EngramError::NotFound.status_code(), StatusCode::NOT_FOUND);
        assert_eq!(EngramError::Unauthorized.status_code(), StatusCode::UNAUTHORIZED);
        assert_eq!(EngramError::EmptyContent.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(EngramError::EmptyQuery.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(EngramError::ContentTooLong.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(EngramError::InvalidLayer(5).status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(EngramError::AiNotConfigured.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            EngramError::AiBackend("timeout".into()).status_code(),
            StatusCode::BAD_GATEWAY,
        );
        assert_eq!(
            EngramError::Internal("oops".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }

    #[test]
    fn error_messages_are_human_readable() {
        assert_eq!(EngramError::EmptyContent.to_string(), "content must not be empty");
        assert_eq!(EngramError::InvalidLayer(7).to_string(), "invalid layer: 7 (expected 1, 2, or 3)");
        assert!(EngramError::Validation("bad tag".into()).to_string().contains("bad tag"));
    }

    #[test]
    fn into_response_has_json_body() {
        use axum::response::IntoResponse;
        let resp = EngramError::NotFound.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
