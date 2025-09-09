'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, Settings, Activity, Brain, History, Search, Plus, Trash2, Stethoscope, Pill, Utensils, AlertTriangle, FileText, User, Calendar, Shield, TrendingUp, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { marked } from 'marked'
// Medical floating particles background
function MedicalParticles() {
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    const newParticles = [];
    for (let i = 0; i < 40; i++) {
      newParticles.push({
        id: i,
        initialX: Math.random() * 100,
        initialY: Math.random() * 100,
        animateX: Math.random() * 100,
        animateY: Math.random() * 100,
        duration: Math.random() * 30 + 25,
        delay: Math.random() * 12,
      });
    }
    setParticles(newParticles);
  }, []);

  if (particles.length === 0) {
    return <div className="absolute inset-0 overflow-hidden" />;
  }

  return (
    <div className="absolute inset-0 overflow-hidden">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute w-1.5 h-1.5 bg-blue-400 rounded-full opacity-40"
          style={{
            left: particle.initialX + '%',
            top: particle.initialY + '%',
          }}
          animate={{
            x: [(particle.animateX - particle.initialX) + 'vw',
                 (particle.initialX - particle.animateX) + 'vw',
                (particle.animateX - particle.initialX) + 'vw'],
            y: [(particle.animateY - particle.initialY) + 'vh',
                (particle.initialY - particle.animateY) + 'vh',
                 (particle.animateY - particle.initialY) + 'vh'],
            scale: [1, 1.2, 1, 0.8, 1],
            opacity: [0.2, 0.6, 0.3, 0.8, 0.2],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: "linear",
            delay: particle.delay,
          }}
        />
      ))}
    </div>
  );
}

// Confidence Score Display Component
function ConfidenceDisplay({ confidence }) {
  const getConfidenceColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'high': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'moderate': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'low': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getConfidenceIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'high': return <CheckCircle className="w-4 h-4" />;
      case 'moderate': return <AlertCircle className="w-4 h-4" />;
      case 'low': return <AlertTriangle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="mb-3 p-3 rounded-lg bg-slate-800/30 border border-slate-700/50"
    >
      <div className="flex items-center justify-between mb-2">
        <div className={`flex items-center gap-2 px-2 py-1 rounded-full border ${getConfidenceColor(confidence.confidence_level)}`}>
          {getConfidenceIcon(confidence.confidence_level)}
          <span className="text-xs font-medium">{confidence.confidence_level} Confidence</span>
        </div>
        <div className="text-sm text-gray-400">
          Overall: {(confidence.overall_confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-400">Medical Validity:</span>
          <span className="text-blue-600">{(confidence.component_scores.medical_validity * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Safety Score:</span>
          <span className="text-green-400">{(confidence.component_scores.safety_score * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Model Confidence:</span>
          <span className="text-yellow-400">{(confidence.component_scores.model_confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Coherence:</span>
          <span className="text-purple-400">{(confidence.component_scores.response_coherence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </motion.div>
  );
}

// Safety Badge Component
function SafetyBadge({ safety }) {
  // Add a check to ensure 'safety' exists and 'safety.is_safe' is a boolean
  if (!safety || typeof safety.is_safe !== 'boolean') {
    // Optionally return null or a loading state if safety data is not ready
    return null; // Or <div className="text-gray-500">Loading safety info...</div>;
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
        safety.is_safe
           ? 'bg-green-500/20 border border-green-500/30 text-green-400'
          : 'bg-red-500/20 border border-red-500/30 text-red-400'
      }`}
    >
      {safety.is_safe ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
      {safety.message}
    </motion.div>
  );
}

// Medical Context Component
function MedicalContext({ medicalContext, domain }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-4 p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
    >
      <div className="flex items-center gap-2 mb-3">
        <Brain className="w-4 h-4 text-blue-600" />
        <span className="text-sm font-medium text-white capitalize">{domain} Clinical Context</span>
      </div>

      {medicalContext.safety_considerations?.length > 0 && (
        <div className="mb-3">
          <h4 className="text-xs font-medium text-gray-300 mb-2 flex items-center gap-1">
            <Shield className="w-3 h-3" />
            Safety Considerations
          </h4>
          <ul className="space-y-1">
            {medicalContext.safety_considerations.map((consideration, index) => (
              <li key={index} className="text-xs text-gray-400 flex items-start gap-2">
                <div className="w-1 h-1 bg-red-400 rounded-full mt-2 flex-shrink-0" />
                {consideration}
              </li>
            ))}
          </ul>
        </div>
      )}

      {medicalContext.follow_up_recommendations?.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-300 mb-2 flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            Follow-up Recommendations
          </h4>
          <ul className="space-y-1">
            {medicalContext.follow_up_recommendations.map((recommendation, index) => (
              <li key={index} className="text-xs text-gray-400 flex items-start gap-2">
                <div className="w-1 h-1 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                {recommendation}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}

// Enhanced Clinical Message Component
function ClinicalMessage({ message, isUser, apiResponse = null, messageType = 'general', delay = 0 }) {
  const getMessageIcon = () => {
    if (apiResponse?.domain) {
      switch (apiResponse.domain) {
        case 'prescription': return <Pill className="w-4 h-4 text-amber-800" />;
        case 'nutrition': return <Utensils className="w-4 h-4 text-orange-500" />;
        case 'diagnosis': return <Stethoscope className="w-4 h-4 text-blue-500" />;
        case 'safety': return <AlertTriangle className="w-4 h-4 text-red-500" />;
        default: return <Brain className="w-4 h-4 text-blue-500" />;
      }
    }
    switch (messageType) {
      case 'prescription': return <Pill className="w-4 h-4 text-amber-800" />;
      case 'diet': return <Utensils className="w-4 h-4 text-orange-500" />;
      case 'diagnosis': return <Stethoscope className="w-4 h-4 text-blue-500" />;
      case 'alert': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      default: return <Brain className="w-4 h-4 text-blue-500" />;
    }
  };

  const getMessageBorder = () => {
    if (apiResponse?.domain) {
      switch (apiResponse.domain) {
        case 'prescription': return 'border-blue-400/30';
        case 'nutrition': return 'border-orange-500/30';
        case 'diagnosis': return 'border-blue-500/30';
        case 'safety': return 'border-red-500/30';
        default: return 'border-blue-500/30';
      }
    }
    switch (messageType) {
      case 'prescription': return 'border-blue-400/30';
      case 'diet': return 'border-orange-500/30';
      case 'diagnosis': return 'border-blue-500/30';
      case 'alert': return 'border-red-500/30';
      default: return 'border-blue-500/30';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}
    >
      <div className={`max-w-[85%] ${isUser ? 'order-2' : 'order-1'}`}>
        <motion.div
          whileHover={{ scale: 1.01, y: -1 }}
          className={`p-4 rounded-2xl backdrop-blur-md border transition-all duration-300 shadow-lg ${
            isUser
               ? 'bg-black/80 slate-900 text-white ml-4 shadow-slate-800/20'
               : `bg-white/10 ${getMessageBorder()} text-gray-100 mr-4 shadow-black/10`
          }`}
        >
          {!isUser && apiResponse && (
            <div className="mb-3">
              <div className="flex items-center gap-2 text-sm opacity-80 mb-2">
                {getMessageIcon()}
                <span className="font-medium capitalize">{apiResponse.domain || messageType} Analysis</span>
                <span className="text-xs text-gray-400">
                  {new Date(apiResponse.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <SafetyBadge safety={apiResponse.safety} />
            </div>
          )}

          {!isUser && apiResponse?.confidence && (
            <ConfidenceDisplay confidence={apiResponse.confidence} />
          )}

          <div dangerouslySetInnerHTML={{ __html:marked.parse(apiResponse?.response || message) }}>
           
          </div>

          {!isUser && apiResponse?.medical_context && (
            <MedicalContext
               medicalContext={apiResponse.medical_context}
               domain={apiResponse.domain}
            />
          )}
        </motion.div>
      </div>
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: delay + 0.2, duration: 0.3 }}
        className={`w-10 h-10 rounded-full flex items-center justify-center shadow-lg ${
          isUser ? 'order-1 bg-slate-700/50 shadow-slate-700/20' : 'order-2 blue-800 shadow-blue-500/10'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-slate-400" />
        ) : (
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Activity className="w-5 h-5 text-blue-600" />
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
}

// Clinical Analysis Typing Indicator
function ClinicalTypingIndicator() {
  const dots = [];
  for (let i = 0; i < 3; i++) {
    dots.push(
      <motion.div
        key={i}
        animate={{
           y: [0, -6, 0],
          opacity: [0.4, 1, 0.4]
         }}
        transition={{
          duration: 1.4,
          repeat: Infinity,
          delay: i * 0.3,
          ease: "easeInOut"
        }}
        className="w-2 h-2 bg-blue-400 rounded-full"
      />
    );
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex justify-start mb-6"
    >
      <motion.div
         animate={{
           boxShadow: [
            '0 0 20px rgba(20, 184, 166, 0.1)',
             '0 0 30px rgba(20, 184, 166, 0.3)',
             '0 0 20px rgba(20, 184, 166, 0.1)'
          ]
         }}
        transition={{ duration: 2.5, repeat: Infinity }}
        className="flex items-center space-x-3 bg-slate-800/60 border border-blue-500/20 p-4 rounded-2xl backdrop-blur-md"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <Activity className="w-4 h-4 text-blue-600" />
        </motion.div>
        <div className="flex space-x-1">
          {dots}
        </div>
        <span className="text-gray-300 text-sm">Analyzing clinical data...</span>
      </motion.div>
    </motion.div>
  );
}

// Quick Action Buttons Component
function QuickActions({ onActionSelect }) {
  const actions = [
    { id: 'prescription', label: 'Prescription Analysis', icon: Pill, color: 'gray' },
    { id: 'diet', label: 'Diet Planning', icon: Utensils, color: 'orange' },
    { id: 'diagnosis', label: 'Diagnostic Support', icon: Stethoscope, color: 'blue' },
    { id: 'alert', label: 'Safety Check', icon: AlertTriangle, color: 'red' }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="mb-6"
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {actions.map((action, index) => (
          <motion.button
            key={action.id}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onActionSelect(action.id)}
            className={`p-3 rounded-xl bg-slate-800/60 border border-${action.color}-500/30 backdrop-blur-md hover:bg-${action.color}-500/10 transition-all duration-300 group`}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <action.icon className={`w-5 h-5 text-${action.color}-400 mx-auto mb-2 group-hover:scale-110 transition-transform duration-300`} />
            <span className="text-xs text-gray-300 block">{action.label}</span>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}

function ClinicalCDSSInterface() {
  const [messages, setMessages] = useState([
    {
       text: "Welcome to the Clinical Decision Support System. I'm here to assist with prescription analysis, diagnostic support, and personalized care planning. How can I help you today?",
       isUser: false,
       messageType: 'general'
     }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMode, setSelectedMode] = useState('general');
  const [apiError, setApiError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // In-memory storage for demo
  const [memoryStorage, setMemoryStorage] = useState({});
  const mockLocalStorage = {
    getItem: (key) => memoryStorage[key] || null,
    setItem: (key, value) => {
      setMemoryStorage(prev => ({ ...prev, [key]: value }));
    }
  };

  useEffect(() => {
    const savedHistory = JSON.parse(mockLocalStorage.getItem('clinicalChatHistory') || '[]');
    setChatHistory(savedHistory);
  }, []);

  // API call function
  const callMedicalAPI = async (query) => {
    try {
      // Replace with your actual API endpoint
      const response = await fetch('http://localhost:8000/api/query', { // Changed to /api/query
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: query, // Changed 'query' to 'question' to match API
          domain: selectedMode !== 'general' ? selectedMode : undefined // Pass domain if not general
        })
      });

      if (!response.ok) {
        // Attempt to parse JSON error message if available
        let errorData = {};
        try {
            errorData = await response.json();
        } catch (jsonError) {
            console.error("Failed to parse error response JSON:", jsonError);
        }
        const errorMessage = errorData.error || `API call failed: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Error:', error);
      // Return mock data for demo purposes if API fails completely
      alert("error")
    }
  };

  const saveCurrentChat = () => {
    if (messages.length <= 1) return; // Don't save if only initial welcome message

    const chatId = currentChatId || Date.now().toString();
    const chatTitle = messages.find(m => m.isUser)?.text.slice(0, 40) + (messages.find(m => m.isUser)?.text.length > 40 ? '...' : '') || 'New Clinical Session'; // Improved title
    const timestamp = new Date().toISOString();

    const chatData = {
      id: chatId,
      title: chatTitle,
      messages: messages,
      timestamp: timestamp,
      mode: selectedMode
    };

    const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
    updatedHistory.unshift(chatData); // Add to the beginning

    setChatHistory(updatedHistory);
    mockLocalStorage.setItem('clinicalChatHistory', JSON.stringify(updatedHistory));
    setCurrentChatId(chatId);
  };

  const loadChat = (chatId) => {
    // Save current chat before loading new one
    saveCurrentChat();

    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setMessages(chat.messages);
      setCurrentChatId(chatId);
      setSelectedMode(chat.mode || 'general');
      setShowHistory(false);
      setApiError(null); // Clear any previous API errors
    }
  };

  const deleteChat = (chatId, e) => {
    e.stopPropagation(); // Prevent loading the chat when deleting
    const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
    setChatHistory(updatedHistory);
    mockLocalStorage.setItem('clinicalChatHistory', JSON.stringify(updatedHistory));

    if (currentChatId === chatId) {
      startNewChat(); // Start a new chat if the active one was deleted
    }
  };

  const startNewChat = () => {
    if (messages.length > 1 || currentChatId) { // Only save if there's actual content or it's an existing chat
      saveCurrentChat();
    }
    setMessages([
      {
       text: "Welcome to the Clinical Decision Support System. I'm here to assist with prescription analysis, diagnostic support, and personalized care planning. How can I help you today?",
       isUser: false,
       messageType: 'general'
     }
    ]);
    setCurrentChatId(null);
    setSelectedMode('general');
    setShowHistory(false);
    setApiError(null);
  };

  const filteredChats = chatHistory.filter(chat =>
    chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.messages.some(msg => msg.text.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleQuickAction = (actionType) => {
    setSelectedMode(actionType);
    const prompts = {
      prescription: "I need help with prescription analysis. Please provide guidance on medication selection and dosing.",
      diet: "I need to create a personalized diet plan based on patient diagnosis and medical history.",
      diagnosis: "I need diagnostic support to analyze symptoms and medical history for potential conditions.",
      alert: "I need to perform a safety check for potential drug interactions or contraindications."
    };

    setInput(prompts[actionType]);
    // Optionally, send immediately after setting input
    // handleSend();
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = input;
    setInput('');
    setApiError(null); // Clear previous API errors

    const newMessages = [...messages, { text: userMessage, isUser: true }];
    setMessages(newMessages);
    setIsTyping(true);

    // Save chat proactively after user sends message
    setTimeout(() => {
      saveCurrentChat();
    }, 100);

    try {
      // Call the actual API
      const apiResponse = await callMedicalAPI(userMessage);

      setIsTyping(false);
      setMessages(prev => {
        const updatedMessages = [...prev, {
           text: apiResponse.response, // Use apiResponse.response for the main text
          isUser: false,
           messageType: apiResponse.domain || selectedMode,
          apiResponse: apiResponse // Pass the full API response for detailed rendering
        }];

        // Save updated chat with new AI response
        setTimeout(() => {
          const chatId = currentChatId || Date.now().toString();
          const chatTitle = updatedMessages.find(m => m.isUser)?.text.slice(0, 40) + (updatedMessages.find(m => m.isUser)?.text.length > 40 ? '...' : '') || 'New Clinical Session';
          const timestamp = new Date().toISOString();

          const chatData = {
            id: chatId,
            title: chatTitle,
            messages: updatedMessages,
            timestamp: timestamp,
            mode: selectedMode
          };
          const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
          updatedHistory.unshift(chatData);

          setChatHistory(updatedHistory);
          mockLocalStorage.setItem('clinicalChatHistory', JSON.stringify(updatedHistory));
          setCurrentChatId(chatId);
        }, 100);

        return updatedMessages;
      });
    } catch (error) {
      setIsTyping(false);
      // Set the API error message to display to the user
      setApiError(error.message || 'Failed to get response from medical API. Please try again.');
      console.error('Error calling API:', error);

      // Add an error message to the chat
      setMessages(prev => [
        ...prev,
        {
          text: `⚠️ Error: ${error.message || 'Could not connect to the AI system. Please try again.'}`,
          isUser: false,
          messageType: 'alert'
        }
      ]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleListening = () => {
    setIsListening(!isListening);
    if (!isListening) {
      // Simulate stopping listening after a short period
      setTimeout(() => setIsListening(false), 4000);
    }
    // In a real app, you would integrate Web Speech API here
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-black relative overflow-hidden">
      {/* Medical Background */}
      <div className="absolute inset-0">
        <MedicalParticles />
      </div>

      {/* Medical Grid Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px'
        }} />
      </div>

      {/* Chat History Sidebar */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ x: -400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -400, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="fixed left-0 top-0 h-full w-80 bg-slate-900/95 backdrop-blur-xl border-r border-slate-700/50 z-50 flex flex-col"
          >
            <div className="p-6 border-b border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <History className="w-5 h-5 text-blue-500" />
                  Clinical Sessions
                </h2>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={startNewChat}
                  className="p-2 rounded-full blue-800 text-blue-600 hover:bg-blue-500/30 transition-all duration-300"
                >
                  <Plus className="w-4 h-4" />
                </motion.button>
              </div>

              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search sessions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-slate-800/60 border slate-900 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500/50"
                />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {filteredChats.length === 0 ? (
                <div className="text-center text-gray-400 mt-8">
                  <p className="mb-4">No sessions found.</p>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={startNewChat}
                    className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 rounded-lg text-white font-medium hover:bg-blue-700 transition-colors duration-300"
                  >
                    <Plus className="w-4 h-4" /> Start New Chat
                  </motion.button>
                </div>
              ) : (
                filteredChats.map((chat) => (
                  <motion.div
                    key={chat.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    onClick={() => loadChat(chat.id)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors duration-200 group relative ${
                      currentChatId === chat.id ? 'bg-blue-500/30 border border-blue-500/50' : 'bg-slate-800/40 hover:bg-slate-700/50 border border-slate-700/40'
                    }`}
                  >
                    <h3 className="text-sm font-medium text-white mb-1 truncate">{chat.title}</h3>
                    <p className="text-xs text-gray-400 truncate">{new Date(chat.timestamp).toLocaleString()}</p>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={(e) => deleteChat(chat.id, e)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-full text-gray-400 hover:text-red-400 hover:bg-slate-700/70 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    >
                      <Trash2 className="w-4 h-4" />
                    </motion.button>
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Interface */}
      <div className={`relative z-10 flex flex-col min-h-screen transition-all duration-300 ${showHistory ? 'ml-80' : 'ml-0'}`}>
        {/* Header */}
        <div className="p-6 bg-slate-900/90 backdrop-blur-xl border-b border-slate-700/50 flex items-center justify-between shadow-lg">
          <div className="flex items-center gap-4">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setShowHistory(!showHistory)}
              className="p-2 rounded-full bg-slate-800/60 text-blue-600 hover:bg-slate-700/50 transition-all duration-300"
            >
              <History className="w-5 h-5" />
            </motion.button>
            <img src='/logo_1.png' style={{"width":'100px'}}></img>
          </div>
          <div className="flex items-center gap-3">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={startNewChat}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-600/30 text-green-400 border border-green-500/40 hover:bg-green-600/40 transition-colors duration-300 text-sm"
            >
              <Plus className="w-4 h-4" /> New Chat
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="p-2 rounded-full bg-slate-800/60 text-gray-400 hover:bg-slate-700/50 transition-all duration-300"
              title="Settings"
            >
              <Settings className="w-5 h-5" />
            </motion.button>
          </div>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900">
          {messages.map((msg, index) => (
            <ClinicalMessage
              key={index}
              message={msg.text}
              isUser={msg.isUser}
              apiResponse={msg.apiResponse}
              messageType={msg.messageType}
              delay={0.05 * index}
            />
          ))}
          {isTyping && <ClinicalTypingIndicator />}
          {apiError && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-center my-4"
              >
                <div className="bg-red-500/20 text-red-400 border border-red-500/30 px-4 py-2 rounded-lg text-sm flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  {apiError}
                </div>
              </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions and Input Area */}
        <div className="p-6 bg-slate-900/90 backdrop-blur-xl border-t border-slate-700/50 shadow-lg">
          {messages.length === 1 && !isTyping && ( // Show quick actions only initially
            <QuickActions onActionSelect={handleQuickAction} />
          )}

          <div className="relative flex items-center bg-slate-800/60 border slate-900 rounded-xl shadow-lg">
            <textarea
              ref={useRef(null)} // Added a ref for textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isListening ? "Listening..." : "Type your clinical query or select a quick action..."}
              rows={1}
              className="flex-1 p-4 bg-transparent text-white placeholder-gray-400 resize-none outline-none focus:ring-0 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-800"
              style={{ maxHeight: '150px' }} // Limit textarea height
            />
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleListening}
              className={`p-3 rounded-full mr-2 transition-all duration-300 ${
                isListening ? 'bg-red-500/30 text-red-400' : 'blue-800 text-blue-600 hover:bg-blue-500/30'
              }`}
            >
              <Mic className="w-5 h-5" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleSend}
              className="p-3 rounded-full bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-all duration-300 mr-2"
              disabled={!input.trim() || isTyping}
            >
              <Send className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ClinicalCDSSInterface;