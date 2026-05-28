use std::collections::BTreeMap;

use ann::tensor::backend::Backend;

use crate::{RetrievalError, SearchHit, encode_pose_features, search_index};

use super::{
    RetrievalService, assets,
    http::{HttpRequest, multipart_boundary, parse_multipart, parse_top_k, parse_urlencoded},
    live::{LiveSearchResponse, live_search_response},
};

pub(super) fn sample_search_from_query(
    service: &RetrievalService<impl Backend>,
    query: &BTreeMap<String, String>,
) -> Result<(Vec<SearchHit>, String, usize), RetrievalError> {
    let top_k = parse_top_k(query.get("k"), service.default_top_k);
    if let Some(example) = query
        .get("example")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let path = assets::example_image_path(example)
            .ok_or_else(|| RetrievalError::InvalidData("invalid example image".to_string()))?;
        let features = service
            .pose_estimator
            .estimate_pose_features_from_path(&path)?;
        let embedding = encode_pose_features(&service.model, &features, &service.device)?;
        let hits = search_index(&service.index, &embedding, top_k)?;
        return Ok((hits, format!("example {example}"), top_k));
    }

    let (pair, source) = if let Some(sample) = query.get("sample") {
        let sample = sample.parse::<usize>().map_err(|_| {
            RetrievalError::InvalidData("sample must be a non-negative integer".to_string())
        })?;
        let pair = service.dataset.pairs().get(sample).ok_or_else(|| {
            RetrievalError::InvalidData(format!("sample index {sample} out of range"))
        })?;
        (pair, format!("sample #{sample} {}", pair.id))
    } else {
        let persona = query
            .get("persona")
            .ok_or_else(|| RetrievalError::InvalidData("missing persona".to_string()))
            .and_then(|value| normalize_persona_query(value))?;
        let id = query
            .get("id")
            .ok_or_else(|| RetrievalError::InvalidData("missing id".to_string()))
            .and_then(|value| normalize_id_query(value))?;
        let pair = service
            .dataset
            .pairs()
            .iter()
            .find(|pair| pair.persona == persona && pair.id == id)
            .ok_or_else(|| {
                RetrievalError::InvalidData(format!("no data pair found for {persona} id {id}"))
            })?;
        (pair, format!("{} {}", pair.persona, pair.id))
    };
    let features = service
        .pose_estimator
        .estimate_pose_features_from_path(&pair.image_path)?;
    let embedding = encode_pose_features(&service.model, &features, &service.device)?;
    let hits = search_index(&service.index, &embedding, top_k)?;
    Ok((hits, source, top_k))
}

pub(super) fn upload_search(
    service: &RetrievalService<impl Backend>,
    request: &HttpRequest,
) -> Result<(Vec<SearchHit>, String, usize), RetrievalError> {
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");

    if let Some(boundary) = multipart_boundary(content_type) {
        let form = parse_multipart(&request.body, &boundary)?;
        let top_k = parse_top_k(form.fields.get("k"), service.default_top_k);
        if form
            .fields
            .get("sample")
            .is_some_and(|value| !value.trim().is_empty())
            || form
                .fields
                .get("persona")
                .is_some_and(|value| !value.trim().is_empty())
            || form
                .fields
                .get("id")
                .is_some_and(|value| !value.trim().is_empty())
        {
            let mut query = BTreeMap::new();
            if let Some(sample) = form.fields.get("sample") {
                query.insert("sample".to_string(), sample.clone());
            }
            if let Some(persona) = form.fields.get("persona") {
                query.insert("persona".to_string(), persona.clone());
            }
            if let Some(id) = form.fields.get("id") {
                query.insert("id".to_string(), id.clone());
            }
            query.insert("k".to_string(), top_k.to_string());
            return sample_search_from_query(service, &query);
        }

        let image = form.files.get("image").ok_or_else(|| {
            RetrievalError::InvalidData("upload an image or provide persona and id".to_string())
        })?;
        if image.is_empty() {
            return Err(RetrievalError::InvalidData(
                "uploaded image is empty".to_string(),
            ));
        }
        let features = service
            .pose_estimator
            .estimate_pose_features_from_bytes(image)?;
        let embedding = encode_pose_features(&service.model, &features, &service.device)?;
        let hits = search_index(&service.index, &embedding, top_k)?;
        return Ok((hits, upload_source_label(form.fields.get("source")), top_k));
    }

    let body = String::from_utf8_lossy(&request.body);
    let form = parse_urlencoded(&body);
    sample_search_from_query(service, &form)
}

fn upload_source_label(value: Option<&String>) -> String {
    let Some(value) = value else {
        return "uploaded image".to_string();
    };
    let label = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if label.is_empty() {
        "uploaded image".to_string()
    } else {
        label.chars().take(80).collect()
    }
}

fn normalize_persona_query(value: &str) -> Result<String, RetrievalError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(RetrievalError::InvalidData(
            "persona is required".to_string(),
        ));
    }
    if value.starts_with("persona_") {
        return Ok(value.to_string());
    }
    let number = value.parse::<usize>().map_err(|_| {
        RetrievalError::InvalidData("persona must be a positive integer".to_string())
    })?;
    if number == 0 {
        return Err(RetrievalError::InvalidData(
            "persona must be a positive integer".to_string(),
        ));
    }
    Ok(format!("persona_{number:02}"))
}

fn normalize_id_query(value: &str) -> Result<String, RetrievalError> {
    let value = value.trim();
    let value = value.strip_suffix(".avif").unwrap_or(value).trim();
    if value.is_empty() {
        return Err(RetrievalError::InvalidData("id is required".to_string()));
    }
    Ok(value.to_string())
}

pub(super) fn live_search(
    service: &RetrievalService<impl Backend>,
    request: &HttpRequest,
) -> Result<LiveSearchResponse, RetrievalError> {
    let top_k = parse_top_k(request.query.get("k"), service.default_top_k);
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");

    let frame = if let Some(boundary) = multipart_boundary(content_type) {
        let form = parse_multipart(&request.body, &boundary)?;
        form.files
            .get("frame")
            .or_else(|| form.files.get("image"))
            .cloned()
            .ok_or_else(|| {
                RetrievalError::InvalidData(
                    "live frame request must include a frame or image file".to_string(),
                )
            })?
    } else {
        request.body.clone()
    };

    if frame.is_empty() {
        return Err(RetrievalError::InvalidData(
            "live frame request body is empty".to_string(),
        ));
    }

    let features = service
        .pose_estimator
        .estimate_pose_features_from_bytes(&frame)?;
    let embedding = encode_pose_features(&service.model, &features, &service.device)?;
    let hits = search_index(&service.index, &embedding, top_k)?;
    Ok(live_search_response(&hits, top_k))
}

#[cfg(test)]
mod tests {
    use super::{normalize_id_query, normalize_persona_query, upload_source_label};

    #[test]
    fn persona_query_accepts_numbers_and_full_directory_names() {
        assert_eq!(normalize_persona_query("1").unwrap(), "persona_01");
        assert_eq!(normalize_persona_query("01").unwrap(), "persona_01");
        assert_eq!(
            normalize_persona_query("persona_fixture").unwrap(),
            "persona_fixture"
        );
    }

    #[test]
    fn id_query_accepts_stems_or_avif_filenames() {
        assert_eq!(normalize_id_query("0083_u4E02").unwrap(), "0083_u4E02");
        assert_eq!(normalize_id_query("0083_u4E02.avif").unwrap(), "0083_u4E02");
    }

    #[test]
    fn upload_source_label_normalizes_optional_label() {
        let label = "  example   0.avif  ".to_string();

        assert_eq!(upload_source_label(Some(&label)), "example 0.avif");
        assert_eq!(upload_source_label(None), "uploaded image");
    }
}
