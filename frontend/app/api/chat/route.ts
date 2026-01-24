const BACKEND_URL = (process.env.BACKEND_URL ?? "").replace(/\/$/, "");
const SERVICE_TOKEN = process.env.SERVICE_TOKEN;

export async function POST(req: Request) {
  const body = await req.json();
  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");

  // Extract the user query from messages and ensure proper format
  let messages = body.messages || [];
  
  // Find the last user message and ensure it has the correct structure
  const userMessages = messages.filter((msg: any) => msg.role === "user");
  if (userMessages.length === 0) {
    console.error("No user message found in request:", { body });
    return new Response(
      JSON.stringify({ error: "No user message provided" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Ensure the user message has content field
  const lastUserMessage = userMessages[userMessages.length - 1];
  if (!lastUserMessage.content && lastUserMessage.parts) {
    // Extract content from parts if needed (assistant-ui format)
    const textPart = lastUserMessage.parts.find((part: any) => part.type === "text");
    if (textPart) {
      lastUserMessage.content = textPart.text;
    }
  }

  const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(SERVICE_TOKEN ? { "X-Service-Token": SERVICE_TOKEN } : {}),
      ...(userId ? { "x-user-id": userId } : {}),
      ...(userAnonymous ? { "x-user-anonymous": userAnonymous } : {}),
    },
    body: JSON.stringify({
      ...body,
      messages,
      stream: true,
    }),
  });

  return new Response(response.body, {
    status: response.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "x-vercel-ai-ui-message-stream": "v1",
    },
  });
}
