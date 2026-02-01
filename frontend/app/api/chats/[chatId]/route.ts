const BACKEND_URL = (process.env.BACKEND_URL ?? "").replace(/\/$/, "");
const SERVICE_TOKEN = process.env.SERVICE_TOKEN;

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ chatId: string }> },
) {
  const chatId = (await params).chatId;
  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");
  const body = await req.json();

  const response = await fetch(`${BACKEND_URL}/chats/${chatId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...(SERVICE_TOKEN ? { "X-Service-Token": SERVICE_TOKEN } : {}),
      ...(userId ? { "x-user-id": userId } : {}),
      ...(userAnonymous ? { "x-user-anonymous": userAnonymous } : {}),
    },
    body: JSON.stringify(body),
  });

  return new Response(await response.text(), {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ chatId: string }> },
) {
  const chatId = (await params).chatId;
  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");

  const response = await fetch(`${BACKEND_URL}/chats/${chatId}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
      ...(SERVICE_TOKEN ? { "X-Service-Token": SERVICE_TOKEN } : {}),
      ...(userId ? { "x-user-id": userId } : {}),
      ...(userAnonymous ? { "x-user-anonymous": userAnonymous } : {}),
    },
  });

  return new Response(await response.text(), {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}
