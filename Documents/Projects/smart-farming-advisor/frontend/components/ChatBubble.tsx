type Props = {
  role: "user" | "assistant";
  text: string;
};

export default function ChatBubble({ role, text }: Props) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-lg rounded-2xl px-4 py-3 text-sm shadow-soft ${
          isUser ? "bg-leaf-600 text-white" : "bg-white text-slate-700"
        }`}
      >
        {text}
      </div>
    </div>
  );
}
