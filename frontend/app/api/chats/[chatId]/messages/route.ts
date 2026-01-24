const BACKEND_URL = (process.env.BACKEND_URL ?? "").replace(/\/$/, "");
const SERVICE_TOKEN = process.env.SERVICE_TOKEN;

export async function GET(
  req: Request,
  { params }: { params: Promise<{ chatId: string }> },
) {
  const chatId = (await params).chatId;

  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");

  const response = await fetch(`${BACKEND_URL}/chats/${chatId}/messages`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      ...(SERVICE_TOKEN ? { "X-Service-Token": SERVICE_TOKEN } : {}),
      ...(userId ? { "x-user-id": userId } : {}),
      ...(userAnonymous ? { "x-user-anonymous": userAnonymous } : {}),
    },
  });

  const bodyText = await response.text();

  try {
    JSON.parse(bodyText);
  } catch (err) {
    console.warn("[chat-messages] failed to parse response", {
      chatId,
      status: response.status,
      error: err,
    });
  }

  return new Response(bodyText, {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}
