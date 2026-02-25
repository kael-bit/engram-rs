use engram::error::EngramError;
use axum::http::StatusCode;

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
