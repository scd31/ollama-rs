use std::sync::Arc;

use ollama_rs::{
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        functions::{tools::Tool, DDGSearcher, Scraper},
    },
    Ollama,
};
use tokio::io::{stdout, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let mut stdout = stdout();
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Scraper::new()), Arc::new(DDGSearcher::new())];

    let mut history = vec![];
    loop {
        if history
            .last()
            .map(|h: &ChatMessage| h.role != MessageRole::Tool)
            .unwrap_or(true)
        {
            stdout.write_all(b"\n> ").await?;
            stdout.flush().await?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            let input = input.trim_end();
            if input.eq_ignore_ascii_case("exit") {
                break;
            }

            let user_message = ChatMessage::user(input.to_string());
            history.push(user_message);
        }

        let result = ollama
            .send_chat_messages(
                ChatMessageRequest::new(
                    "qwen2.5:32b".to_string(),
                    history.clone(), // TODO can we clean up the clone
                )
                .tools(&tools),
            )
            .await?;

        let msg = result.message.unwrap();

        history.push(msg.clone());

        if msg.tool_calls.is_empty() {
            let assistant_message = &msg.content;
            stdout.write_all(assistant_message.as_bytes()).await?;
            stdout.flush().await?;
        } else {
            for call in msg.tool_calls {
                let tool = tools
                    .iter()
                    .find(|t| t.name() == call.function.name)
                    .unwrap();

                let resp = tool.run(call.function.arguments).await?;

                history.push(ChatMessage::new(MessageRole::Tool, resp));
            }
        }
    }

    // Display whole history of messages
    dbg!(history);

    Ok(())
}
