const BACKEND_URL = (process.env.BACKEND_URL ?? "").replace(/\/$/, "");
const SERVICE_TOKEN = process.env.SERVICE_TOKEN;

export async function GET(req: Request) {
  const userId = req.headers.get("x-user-id");
  const userAnonymous = req.headers.get("x-user-anonymous");

  const response = await fetch(`${BACKEND_URL}/get_chats`, {
    method: "GET",
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
