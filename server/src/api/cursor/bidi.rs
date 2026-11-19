//! Accepts ordered Cursor Bidi append requests and routes them by request_id.
use prost::Message;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{
            events,
            proto::{agent::v1 as agent, aiserver::v1 as ai},
        },
        transport::{TransportParent, TransportRegistry},
    },
    Error, Result,
};

pub struct DecodedAppend {
    pub request_id: String,
    pub seqno: i64,
    pub message: agent::AgentClientMessage,
}

impl DecodedAppend {
    pub fn model_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str())
            .filter(|model| !model.is_empty())
            .or_else(|| {
                request
                    .model_details
                    .as_ref()
                    .map(|model| model.model_id.as_str())
                    .filter(|model| !model.is_empty())
            })
    }

    pub fn conversation_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request.conversation_id.as_deref()
    }

    pub fn is_background_task_completion(&self) -> bool {
        let Some(agent::agent_client_message::Message::RunRequest(request)) =
            self.message.message.as_ref()
        else {
            return false;
        };
        matches!(
            request
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(agent::conversation_action::Action::BackgroundTaskCompletionAction(_))
        )
    }

    pub fn trace_metadata(&self) -> serde_json::Value {
        let Some(message) = self.message.message.as_ref() else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": "empty",
            });
        };
        let agent::agent_client_message::Message::RunRequest(request) = message else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": client_message_type(message),
            });
        };
        let (action_type, history_messages, history_images) = request
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref())
            .map(|action| match action {
                agent::conversation_action::Action::UserMessageAction(action) => {
                    let history = action.conversation_history.as_ref();
                    (
                        "user_message",
                        history.map_or(0, |history| history.messages.len()),
                        history.map_or(0, history_image_count),
                    )
                }
                agent::conversation_action::Action::BackgroundTaskCompletionAction(_) => {
                    ("background_task_completion", 0, 0)
                }
                agent::conversation_action::Action::ExecutePlanAction(_) => ("execute_plan", 0, 0),
                agent::conversation_action::Action::SummarizeAction(_) => ("summarize", 0, 0),
                _ => ("other", 0, 0),
            })
            .unwrap_or(("none", 0, 0));
        let state = request.conversation_state.as_ref();
        serde_json::json!({
            "append_seqno": self.seqno,
            "message_type": "run_request",
            "conversation_id": request.conversation_id,
            "model_id": self.model_id(),
            "action_type": action_type,
            "conversation_history_messages": history_messages,
            "conversation_history_images": history_images,
            "root_message_count": state.map_or(0, |state| state.root_prompt_messages_json.len()),
            "turn_count": state.map_or(0, |state| state.turns.len()),
            "prefetched_blob_count": request.pre_fetched_blobs.len(),
        })
    }
}

fn client_message_type(message: &agent::agent_client_message::Message) -> &'static str {
    use agent::agent_client_message::Message;
    match message {
        Message::RunRequest(_) => "run_request",
        Message::ExecClientMessage(_) => "exec_client_message",
        Message::ExecClientControlMessage(_) => "exec_client_control_message",
        Message::KvClientMessage(_) => "kv_client_message",
        Message::ConversationAction(_) => "conversation_action",
        Message::InteractionResponse(_) => "interaction_response",
        Message::ClientHeartbeat(_) => "client_heartbeat",
        Message::PrewarmRequest(_) => "prewarm_request",
    }
}

fn history_image_count(history: &agent::ConversationHistory) -> usize {
    use agent::{
        conversation_history_message::Message,
        conversation_history_tool_result_content::Content as ToolContent,
        conversation_history_user_content::Content as UserContent,
    };
    history
        .messages
        .iter()
        .map(|message| match message.message.as_ref() {
            Some(Message::User(user)) => user
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(UserContent::Image(_))))
                .count(),
            Some(Message::Tool(tool)) => tool
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(ToolContent::Image(_))))
                .count(),
            _ => 0,
        })
        .sum()
}

pub fn decode(request: &ai::BidiAppendRequest) -> Result<DecodedAppend> {
    let request_id = request
        .request_id
        .as_ref()
        .map(|id| id.request_id.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::Protocol("BidiAppend request_id is required".into()))?;
    Ok(DecodedAppend {
        request_id: request_id.into(),
        seqno: request.append_seqno,
        message: agent::AgentClientMessage::decode(append_payload(request)?.as_slice())?,
    })
}

/// Cursor serializes the same `AgentClientMessage` two ways: clients that keep
/// binary appends off send it hex encoded in `data`, while clients that turn
/// them on send the raw protobuf bytes in `data_binary`. Accept both.
fn append_payload(request: &ai::BidiAppendRequest) -> Result<Vec<u8>> {
    if !request.data_binary.is_empty() {
        return Ok(request.data_binary.clone());
    }
    if request.data.is_empty() {
        return Err(Error::Protocol(
            "BidiAppend contains no AgentClientMessage".into(),
        ));
    }
    hex::decode(&request.data)
        .map_err(|error| Error::Protocol(format!("invalid BidiAppend hex: {error}")))
}

pub async fn append(
    registry: &TransportRegistry,
    request: DecodedAppend,
    parent: Option<TransportParent>,
) -> Result<ai::BidiAppendResponse> {
    let replace_closing = request.model_id().is_some();
    let handle = registry
        .get_or_create_for_append(&request.request_id, replace_closing)
        .await?;
    let _admission = handle.admit()?;
    if let Some(conversation_id) = request.conversation_id() {
        handle.set_conversation_id(conversation_id)?;
    }
    if let Some(parent) = parent {
        handle.set_parent(parent)?;
    }
    if matches!(
        request.message.message.as_ref(),
        Some(agent::agent_client_message::Message::ClientHeartbeat(_))
    ) {
        handle.emit(&events::heartbeat())?;
    }
    handle
        .command(TransportCommand::Append {
            seqno: request.seqno,
            message: Box::new(request.message),
        })
        .await?;
    Ok(ai::BidiAppendResponse {})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_request(model_id: &str) -> agent::AgentClientMessage {
        agent::AgentClientMessage {
            message: Some(agent::agent_client_message::Message::RunRequest(
                agent::AgentRunRequest {
                    requested_model: Some(agent::RequestedModel {
                        model_id: model_id.into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn append_request(data: &str, data_binary: Vec<u8>) -> ai::BidiAppendRequest {
        ai::BidiAppendRequest {
            data: data.into(),
            request_id: Some(ai::BidiRequestId {
                request_id: "request-id".into(),
            }),
            append_seqno: 1,
            data_binary,
        }
    }

    #[test]
    fn hex_payload_decodes_agent_client_message() {
        let message = run_request("plugin:demo/openai/model");
        let decoded = decode(&append_request(
            &hex::encode(message.encode_to_vec()),
            Vec::new(),
        ))
        .expect("hex append decodes");

        assert_eq!(decoded.message, message);
        assert_eq!(decoded.model_id(), Some("plugin:demo/openai/model"));
    }

    #[test]
    fn binary_payload_decodes_agent_client_message() {
        let message = run_request("plugin:demo/openai/model");
        let decoded =
            decode(&append_request("", message.encode_to_vec())).expect("binary append decodes");

        assert_eq!(decoded.request_id, "request-id");
        assert_eq!(decoded.seqno, 1);
        assert_eq!(decoded.message, message);
        assert_eq!(decoded.model_id(), Some("plugin:demo/openai/model"));
    }

    #[test]
    fn binary_payload_matches_hex_payload() {
        let message = run_request("plugin:demo/openai/model");
        let from_hex = decode(&append_request(
            &hex::encode(message.encode_to_vec()),
            Vec::new(),
        ))
        .expect("hex append decodes");
        let from_binary =
            decode(&append_request("", message.encode_to_vec())).expect("binary append decodes");

        assert_eq!(from_hex.message, from_binary.message);
    }

    #[test]
    fn binary_payload_wins_when_both_are_present() {
        let binary = run_request("plugin:demo/binary");
        let hex_encoded = run_request("plugin:demo/hex");
        let decoded = decode(&append_request(
            &hex::encode(hex_encoded.encode_to_vec()),
            binary.encode_to_vec(),
        ))
        .expect("append decodes");

        assert_eq!(decoded.model_id(), Some("plugin:demo/binary"));
    }

    #[test]
    fn append_without_payload_is_rejected() {
        let error = decode_rejection(append_request("", Vec::new()));

        assert!(error.contains("contains no AgentClientMessage"));
    }

    #[test]
    fn malformed_hex_payload_is_rejected() {
        let error = decode_rejection(append_request("zz", Vec::new()));

        assert!(error.contains("invalid BidiAppend hex"));
    }

    #[test]
    fn malformed_binary_payload_is_rejected() {
        let error = decode_rejection(append_request("", vec![0x0a]));

        assert!(error.contains("protobuf decode error"));
    }

    fn decode_rejection(request: ai::BidiAppendRequest) -> String {
        match decode(&request) {
            Ok(decoded) => panic!("append decoded unexpectedly: {}", decoded.request_id),
            Err(error) => error.to_string(),
        }
    }
}
