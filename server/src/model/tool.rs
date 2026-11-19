//! Defines Tool calls, results, and Tool round identities.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ProviderReplayState;

pub fn normalize_tool_name(name: &str) -> String {
    let normalized = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if normalized.is_empty() {
        "_".into()
    } else {
        normalized
    }
}

/// Rewrites integral floating-point numbers (`1.0`) as integers (`1`).
///
/// Protobuf `Struct` values only carry doubles, so tool schema keywords that
/// must be integers (`minLength`, `maxItems`, ...) turn into `1.0` when the
/// schema is converted to JSON, and strict validators reject the request.
/// Values that are already integers, and numbers with a fractional part, are
/// left untouched.
pub fn normalize_json_integers(value: Value) -> Value {
    match value {
        Value::Number(number) if !number.is_i64() && !number.is_u64() => match number.as_f64() {
            Some(float)
                if float.fract() == 0.0 && float >= i64::MIN as f64 && float <= i64::MAX as f64 =>
            {
                Value::Number((float as i64).into())
            }
            _ => Value::Number(number),
        },
        Value::Array(values) => {
            Value::Array(values.into_iter().map(normalize_json_integers).collect())
        }
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, normalize_json_integers(value)))
                .collect(),
        ),
        other => other,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub index: usize,
    pub call_id: String,
    pub model_call_id: String,
    pub name: String,
    pub arguments_text: String,
    pub arguments: Value,
    pub argument_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
    pub image: Option<ToolImageReference>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolImageReference {
    pub blob_id: String,
    pub mime_type: String,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolRoundAssistant {
    pub text: String,
    pub thinking: String,
    pub model_call_id: String,
    pub replay_state: Option<ProviderReplayState>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::normalize_json_integers;

    #[test]
    fn integral_floats_are_rewritten_as_integers() {
        let schema = json!({
            "type": "object",
            "anyOf": [
                {"properties": {"rootPath": {"type": "string", "minLength": 1.0, "maxLength": 4096.0}}}
            ],
            "properties": {"limit": {"type": "number", "minimum": 0.0, "default": 2.5}},
        });

        let normalized = normalize_json_integers(schema);

        assert_eq!(
            normalized,
            json!({
                "type": "object",
                "anyOf": [
                    {"properties": {"rootPath": {"type": "string", "minLength": 1, "maxLength": 4096}}}
                ],
                "properties": {"limit": {"type": "number", "minimum": 0, "default": 2.5}},
            })
        );
    }

    #[test]
    fn non_numbers_and_fractional_numbers_are_untouched() {
        assert_eq!(
            normalize_json_integers(json!({"a": true, "b": null, "c": "1.0", "d": [1.5, -2.5]})),
            json!({"a": true, "b": null, "c": "1.0", "d": [1.5, -2.5]})
        );
    }
}
