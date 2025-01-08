document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const themeSlider = document.getElementById('theme-slider');
    
    // Memory management
    const clearMemoryBtn = document.getElementById('clearMemory');
    const resetSystemBtn = document.getElementById('resetSystem');

    let isProcessing = false;

    // Configure marked options
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });

    // Create custom renderer
    const renderer = new marked.Renderer();

    // Custom link rendering for maps
    renderer.link = function(href, title, text) {
        console.log('Link rendering:', { href, title, text }); // Debug log
        
        // Handle raw object input
        if (typeof href === 'object') {
            console.log('Received object href:', href);
            if (href.href) {
                href = href.href;
            } else if (href.raw) {
                href = href.raw;
            } else {
                console.error('Invalid href object:', href);
                return text || 'Invalid Link';
            }
        }

        // Clean and validate URL
        try {
            href = decodeURIComponent(href).trim();
            new URL(href); // Validate URL format
        } catch (e) {
            console.error('Invalid URL:', href);
            return text || 'Invalid URL';
        }

        // Handle Google Maps links
        if (href.includes('google.com/maps')) {
            return `<div class="map-container">
                <p>Click below to open directions in Google Maps:</p>
                <a href="${href}" target="_blank" class="map-link btn btn-primary">Open in Google Maps</a>
            </div>`;
        }

        // For regular links, ensure we preserve the original text and never use fallbacks
        if (!text || text === 'undefined' || text === '[object Object]') {
            console.error('Invalid link text:', text);
            return `<a href="${href}" ${title ? `title="${title}"` : ''} target="_blank">${title || href}</a>`;
        }
        
        return `<a href="${href}" ${title ? `title="${title}"` : ''} target="_blank">${text}</a>`;
    };

    // Add renderer to marked options
    marked.setOptions({ renderer: renderer });

    // Theme control
    function updateTheme(value) {
        // Convert slider value (0-100) to darkness level (0.1-1)
        const darkness = 0.1 + (value / 100) * 0.9;
        
        // Update background colors
        document.documentElement.style.setProperty('--bg-color', `hsl(222, 47%, ${darkness * 8}%)`);
        document.documentElement.style.setProperty('--chat-window-bg', `hsl(222, 47%, ${darkness * 13}%)`);
        document.documentElement.style.setProperty('--secondary-color', `hsl(222, 47%, ${darkness * 15}%)`);
        document.documentElement.style.setProperty('--accent-color', `hsl(222, 47%, ${darkness * 20}%)`);
        document.documentElement.style.setProperty('--input-bg', `hsl(222, 47%, ${darkness * 15}%)`);
        document.documentElement.style.setProperty('--message-assistant-bg', `hsl(222, 47%, ${darkness * 20}%)`);
        
        // Update border opacity based on darkness
        const borderOpacity = 0.1 + ((100 - value) / 100) * 0.2;
        document.documentElement.style.setProperty('--chat-border', `rgba(59, 130, 246, ${borderOpacity})`);
    }

    themeSlider.addEventListener('input', function() {
        updateTheme(this.value);
    });

    // Initialize theme
    updateTheme(themeSlider.value);

    function createMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        messageDiv.textContent = text;
        return messageDiv;
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Markdown regex patterns for raw text processing
    const markdownPatterns = {
        codeBlock: /```[\s\S]*?```/g,          // Code blocks with backticks
        inlineCode: /`[^`]+`/g,                // Inline code
        headers: /^#{1,6}\s+.+$/gm,            // Headers
        lists: /^[\s]*[-*+]\s+.+$/gm,          // Unordered lists
        numberedLists: /^[\s]*\d+\.\s+.+$/gm,  // Ordered lists
        blockquotes: /^>\s+.+$/gm,             // Blockquotes
        emphasis: /[*_]{1,2}[^*_]+[*_]{1,2}/g, // Bold and italic
        links: /\[([^\]]+)\]\(([^)]+)\)/g,     // Links
        images: /!\[([^\]]+)\]\(([^)]+)\)/g,   // Images
        tables: /\|[^|\r\n]*\|/g,              // Tables
        horizontalRules: /^[-*_]{3,}$/gm       // Horizontal rules
    };

    async function sendMessage() {
        if (isProcessing || !userInput.value.trim()) return;
        
        isProcessing = true;
        sendButton.disabled = true;
        userInput.disabled = true;
        
        // Create and display user message
        const userMessage = userInput.value.trim();
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'message user-message';
        userMessageDiv.textContent = userMessage;
        messagesContainer.appendChild(userMessageDiv);
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Create assistant message container
        const assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.className = 'message assistant-message';
        messagesContainer.appendChild(assistantMessageDiv);
        
        try {
            const response = await fetch('/chat', {  // <-- Fixed: Use absolute path
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantResponse = '';
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            if (data.token) {
                                assistantResponse += data.token;
                                assistantMessageDiv.innerHTML = marked.parse(assistantResponse);
                                scrollToBottom();
                            }
                        } catch (e) {
                            console.error('Error parsing chunk:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            assistantMessageDiv.textContent = `Error: ${error.message}`;
        } finally {
            isProcessing = false;
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Memory management functions
    clearMemoryBtn.addEventListener('click', async () => {
        if (isProcessing) return;
        isProcessing = true;
        clearMemoryBtn.disabled = true;

        try {
            const response = await fetch('/clear_memory', {
                method: 'POST'
            });
            const result = await response.json();
            if (result.status === 'success') {
                messagesContainer.innerHTML = '';
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant-message';
                messageDiv.textContent = 'Memory cleared successfully';
                messagesContainer.appendChild(messageDiv);
            }
        } catch (error) {
            console.error('Failed to clear memory:', error);
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message';
            messageDiv.textContent = `Error: ${error.message || 'Failed to clear memory'}`;
            messagesContainer.appendChild(messageDiv);
        } finally {
            isProcessing = false;
            clearMemoryBtn.disabled = false;
            scrollToBottom();
        }
    });

    resetSystemBtn.addEventListener('click', async () => {
        if (isProcessing || !confirm('This will reset the entire system state. Are you sure?')) return;
        isProcessing = true;
        resetSystemBtn.disabled = true;

        try {
            const response = await fetch('/flush_redis', {
                method: 'POST'
            });
            const result = await response.json();
            if (result.status === 'success') {
                messagesContainer.innerHTML = '';
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant-message';
                messageDiv.textContent = 'System reset successfully';
                messagesContainer.appendChild(messageDiv);
                setTimeout(() => location.reload(), 1000);
            }
        } catch (error) {
            console.error('Failed to reset system:', error);
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message';
            messageDiv.textContent = `Error: ${error.message || 'Failed to reset system'}`;
            messagesContainer.appendChild(messageDiv);
        } finally {
            isProcessing = false;
            resetSystemBtn.disabled = false;
            scrollToBottom();
        }
    });
}); 