import LoginForm from "@/components/LoginForm";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-bold mb-6 text-center">Deepfake Detection</h1>
        <LoginForm />
      </div>
    </main>
  );
}
