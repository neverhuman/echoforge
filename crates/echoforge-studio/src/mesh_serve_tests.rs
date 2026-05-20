use super::*;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

#[test]
fn canonical_list_has_five_primitive_ids_in_stable_order() {
    let resp = MeshListResponse::canonical();
    let ids: Vec<&str> = resp
        .primitives
        .iter()
        .map(|p| p.primitive_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["sphere", "plate", "cylinder", "dihedral", "trihedral"]
    );
    for descriptor in &resp.primitives {
        assert!(
            !descriptor.defaults.is_empty(),
            "defaults missing for {}",
            descriptor.primitive_id
        );
    }
}

#[test]
fn build_mesh_applies_defaults_when_params_empty() {
    for primitive in PRIMITIVE_IDS {
        let mesh = match build_mesh(primitive, &MeshParams::default()) {
            Ok(m) => m,
            Err(e) => panic!("primitive {primitive} should build with defaults: {e}"),
        };
        assert_eq!(mesh.primitive_id, primitive);
        assert!(
            mesh.triangle_count() > 0,
            "{primitive} mesh must be non-empty"
        );
        assert!(mesh.all_finite(), "{primitive} mesh must be finite");
    }
}

#[test]
fn build_mesh_rejects_invalid_inputs() {
    let bad_radius = MeshParams {
        radius: Some(0.0),
        ..Default::default()
    };
    assert!(matches!(
        build_mesh("sphere", &bad_radius),
        Err(MeshServeError::InvalidParameter { name: "radius", .. })
    ));

    let bad_n_lon = MeshParams {
        n_lon: Some(2),
        ..Default::default()
    };
    assert!(matches!(
        build_mesh("sphere", &bad_n_lon),
        Err(MeshServeError::InvalidParameter { name: "n_lon", .. })
    ));

    let oversize = MeshParams {
        n_lat: Some(MAX_SAMPLES + 1),
        ..Default::default()
    };
    assert!(matches!(
        build_mesh("sphere", &oversize),
        Err(MeshServeError::InvalidParameter { name: "n_lat", .. })
    ));

    assert!(matches!(
        build_mesh("not_a_primitive", &MeshParams::default()),
        Err(MeshServeError::UnknownPrimitive(_))
    ));
}

#[test]
fn build_mesh_honors_explicit_overrides() {
    let params = MeshParams {
        radius: Some(2.5),
        n_lat: Some(4),
        n_lon: Some(6),
        ..Default::default()
    };
    let mesh = build_mesh("sphere", &params).expect("ok");
    assert_eq!(mesh.dimensions, vec![2.5]);
    // (n_lat - 1) * n_lon * 2 = 3 * 6 * 2 = 36 triangles.
    assert_eq!(mesh.triangle_count(), 36);
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt")
}

#[test]
fn list_endpoint_returns_five_primitive_ids() {
    let app = mesh_router();
    let response = rt().block_on(async {
        app.oneshot(
            Request::builder()
                .uri("/api/mesh/list")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    });
    assert_eq!(response.status(), StatusCode::OK);
    let body = rt()
        .block_on(async { axum::body::to_bytes(response.into_body(), usize::MAX).await })
        .expect("body");
    let parsed: MeshListResponse = serde_json::from_slice(&body).expect("parse");
    let ids: Vec<String> = parsed
        .primitives
        .into_iter()
        .map(|p| p.primitive_id)
        .collect();
    assert_eq!(
        ids,
        vec![
            "sphere".to_string(),
            "plate".to_string(),
            "cylinder".to_string(),
            "dihedral".to_string(),
            "trihedral".to_string(),
        ]
    );
}

#[test]
fn sphere_endpoint_returns_binary_stl_with_expected_content_type() {
    let app = mesh_router();
    let response = rt().block_on(async {
        app.oneshot(
            Request::builder()
                .uri("/api/mesh/sphere?radius=1.0")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    });
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.contains("stl") || content_type.contains("octet-stream"),
        "unexpected Content-Type: {content_type}"
    );
    let triangle_header = response
        .headers()
        .get("x-echoforge-triangle-count")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let triangle_count: usize = triangle_header.parse().expect("triangle-count header");
    assert!(triangle_count > 0, "triangle count header must be > 0");
    let body = rt()
        .block_on(async { axum::body::to_bytes(response.into_body(), usize::MAX).await })
        .expect("body");
    // Binary STL: 80-byte header + 4-byte count + 50 bytes/tri.
    let expected_len = 84 + triangle_count * 50;
    assert_eq!(body.len(), expected_len, "STL length mismatch");
    // Triangle count word at offset 80..84 must match the header.
    let header_count = u32::from_le_bytes(body[80..84].try_into().expect("4 bytes"));
    assert_eq!(header_count as usize, triangle_count);
}

#[test]
fn unknown_primitive_returns_404() {
    let app = mesh_router();
    let response = rt().block_on(async {
        app.oneshot(
            Request::builder()
                .uri("/api/mesh/unknown")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    });
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = rt()
        .block_on(async { axum::body::to_bytes(response.into_body(), usize::MAX).await })
        .expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["error"].as_str(), Some("unknown_primitive"));
}

#[test]
fn invalid_parameter_returns_400() {
    let app = mesh_router();
    let response = rt().block_on(async {
        app.oneshot(
            Request::builder()
                .uri("/api/mesh/sphere?radius=-1")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    });
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = rt()
        .block_on(async { axum::body::to_bytes(response.into_body(), usize::MAX).await })
        .expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["error"].as_str(), Some("invalid_parameter"));
}

#[test]
fn all_primitives_round_trip_via_http() {
    let app = mesh_router();
    for primitive in PRIMITIVE_IDS {
        let response = rt().block_on(async {
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/api/mesh/{primitive}"))
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response")
        });
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "primitive {primitive} HTTP status"
        );
        let body = rt()
            .block_on(async { axum::body::to_bytes(response.into_body(), usize::MAX).await })
            .expect("body");
        assert!(
            body.len() > 84,
            "primitive {primitive} STL body too small ({} bytes)",
            body.len()
        );
    }
}
