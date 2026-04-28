import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem("chats");
    return saved ? JSON.parse(saved) : [];
  });
  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Save chats to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("chats", JSON.stringify(chats));
  }, [chats]);

  // Auto scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats, activeChatId]);

  // Get active chat
  const activeChat = chats.find(c => c.id === activeChatId);

  // Create new chat
  const createNewChat = () => {
    setActiveChatId(null);
  };

  // Handle file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${process.env.REACT_APP_API_URL || "http://localhost:8000"}/upload`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" }
        }
      );

      const newChat = {
        id: response.data.session_id,
        title: file.name,
        fileName: file.name,
        messages: [
          {
            role: "ai",
            text: `Document "${file.name}" uploaded! Ask me anything about it.`
          }
        ],
        createdAt: new Date().toISOString()
      };

      setChats(prev => [newChat, ...prev]);
      setActiveChatId(newChat.id);

    } catch (error) {
      alert("Upload failed. Make sure backend is running.");
    }

    setUploading(false);
    e.target.value = "";
  };

  // Send message
  const sendMessage = async () => {
    if (!input.trim() || !activeChatId) return;

    const userMessage = { role: "user", text: input };

    // Add user message
    setChats(prev => prev.map(chat =>
      chat.id === activeChatId
        ? { ...chat, messages: [...chat.messages, userMessage] }
        : chat
    ));

    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL || "http://localhost:8000"}/ask`,
        {
          session_id: activeChatId,
          question: input
        }
      );

      const aiMessage = {
        role: "ai",
        text: response.data.answer
      };

      setChats(prev => prev.map(chat =>
        chat.id === activeChatId
          ? { ...chat, messages: [...chat.messages, aiMessage] }
          : chat
      ));

    } catch (error) {
      setChats(prev => prev.map(chat =>
        chat.id === activeChatId
          ? {
            ...chat, messages: [...chat.messages, {
              role: "ai",
              text: "Sorry, something went wrong."
            }]
          }
          : chat
      ));
    }

    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  // Delete chat
  const deleteChat = async (chatId, e) => {
    e.stopPropagation();
    await axios.delete(
      `${process.env.REACT_APP_API_URL || "http://localhost:8000"}/session/${chatId}`
    ).catch(() => {});
    setChats(prev => prev.filter(c => c.id !== chatId));
    if (activeChatId === chatId) setActiveChatId(null);
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>DocChat AI</h2>
          <button
            className="new-chat-btn"
            onClick={createNewChat}
          >
            + New Chat
          </button>
        </div>

        <div className="chat-list">
          {chats.length === 0 && (
            <p className="no-chats">No chats yet. Upload a document!</p>
          )}
          {chats.map(chat => (
            <div
              key={chat.id}
              className={`chat-item ${activeChatId === chat.id ? "active" : ""}`}
              onClick={() => setActiveChatId(chat.id)}
            >
              <span className="chat-icon">📄</span>
              <div className="chat-info">
                <p className="chat-title">{chat.title}</p>
                <p className="chat-subtitle">
                  {chat.messages.length - 1} messages
                </p>
              </div>
              <button
                className="delete-btn"
                onClick={(e) => deleteChat(chat.id, e)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main Area */}
      <div className="main">
        {!activeChatId ? (
          /* Welcome Screen */
          <div className="welcome">
            <div className="welcome-content">
              <h1>DocChat AI</h1>
              <p>Upload a document and ask questions about it</p>
              <div className="upload-area"
                onClick={() => fileInputRef.current.click()}
              >
                {uploading ? (
                  <p>Uploading...</p>
                ) : (
                  <>
                    <span className="upload-icon">📁</span>
                    <p>Click to upload document</p>
                    <span className="upload-hint">
                      Supports PDF, TXT, DOCX
                    </span>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.docx"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
            </div>
          </div>
        ) : (
          /* Chat Screen */
          <>
            {/* Chat Header */}
            <div className="chat-header">
              <span>📄</span>
              <p>{activeChat?.fileName}</p>
              <button
                className="upload-new-btn"
                onClick={() => fileInputRef.current.click()}
              >
                Upload New
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.docx"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
            </div>

            {/* Messages */}
            <div className="messages">
              {activeChat?.messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="bubble">
                    <span className="role">
                      {msg.role === "ai" ? "🤖 AI" : "👤 You"}
                    </span>
                    <p>{msg.text}</p>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message ai">
                  <div className="bubble">
                    <span className="role">🤖 AI</span>
                    <p className="typing">Thinking...</p>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <div className="input-area">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask anything about the document..."
                disabled={loading}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
              >
                Send
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;