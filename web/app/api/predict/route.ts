const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

export async function POST(request: Request): Promise<Response> {
  const authHeader = request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return new Response(JSON.stringify({ error: "Missing or invalid token" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const formData = await request.formData();
  const proxyForm = new FormData();
  const image = formData.get("image");
  const text = formData.get("text");

  if (image) proxyForm.append("image", image);
  if (text) proxyForm.append("text", text as string);

  let backendRes: Response;
  try {
    backendRes = await fetch(`${PYTHON_BACKEND_URL}/detect`, {
      method: "POST",
      headers: { Authorization: authHeader },
      body: proxyForm,
    });
  } catch {
    return new Response(JSON.stringify({ error: "Backend unavailable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (backendRes.status === 401) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (backendRes.status === 422) {
    const detail = await backendRes.json().catch(() => ({ error: "Unprocessable entity" }));
    return new Response(JSON.stringify(detail), {
      status: 422,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!backendRes.ok) {
    return new Response(JSON.stringify({ error: "Detection failed" }), {
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
