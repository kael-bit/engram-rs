use axum::body::Bytes;
use axum::extract::{FromRequest, Request};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;

/// Like axum's `Json<T>`, but doesn't require `Content-Type: application/json`.
///
/// If the header is present and isn't JSON, rejects with 415.
/// If absent, the body is parsed as JSON anyway.
/// This saves agents from needing `-H 'Content-Type: application/json'` in every curl.
pub struct LenientJson<T>(pub T);

pub struct LenientJsonRejection {
    message: String,
    status: StatusCode,
}

impl IntoResponse for LenientJsonRejection {
    fn into_response(self) -> Response {
        (self.status, self.message).into_response()
    }
}

impl<S, T> FromRequest<S> for LenientJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = LenientJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        if let Some(ct) = req.headers().get(header::CONTENT_TYPE) {
            let ct_str = ct.to_str().unwrap_or("");
            // Accept: missing, application/json, or curl's default (x-www-form-urlencoded).
            // Reject only explicitly wrong types like text/xml, multipart/form-data, etc.
            if !ct_str.contains("application/json")
                && !ct_str.contains("application/x-www-form-urlencoded")
            {
                return Err(LenientJsonRejection {
                    message: format!("Expected application/json, got {ct_str}"),
                    status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
                });
            }
        }

        let bytes = Bytes::from_request(req, state).await.map_err(|e| {
            LenientJsonRejection {
                message: e.to_string(),
                status: StatusCode::BAD_REQUEST,
            }
        })?;

        serde_json::from_slice(&bytes).map(LenientJson).map_err(|e| {
            LenientJsonRejection {
                message: format!("Invalid JSON: {e}"),
                status: StatusCode::BAD_REQUEST,
            }
        })
    }
}
