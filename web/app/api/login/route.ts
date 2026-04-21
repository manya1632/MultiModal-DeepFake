const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

export async function POST(request: Request): Promise<Response> {
  const formData = await request.formData();

  const proxyForm = new FormData();
  const image = formData.get("image");
  const username = formData.get("username");
  const password = formData.get("password");

  if (image) proxyForm.append("image", image);
  if (username) proxyForm.append("username", username as string);
  if (password) proxyForm.append("password", password as string);

  let backendRes: Response;
  try {
    backendRes = await fetch(`${PYTHON_BACKEND_URL}/auth/login`, {
      method: "POST",
      body: proxyForm,
    });
  } catch {
    return new Response(JSON.stringify({ error: "Backend unavailable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (backendRes.status === 401) {
    return new Response(JSON.stringify({ error: "Invalid credentials" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!backendRes.ok) {
    return new Response(JSON.stringify({ error: "Login failed" }), {
      status: backendRes.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  const data = await backendRes.json();
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
