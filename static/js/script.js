// MathJax Configuration
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true,
        packages: ['base', 'ams', 'noerrors', 'noundefined']
    },
    svg: {
        fontCache: 'global',
        scale: 1,                  // global scaling factor for all expressions
        minScale: .5,             // smallest scaling factor to use
    },
    options: {
        enableMenu: false,
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    },
    startup: {
        typeset: false
    }
};

document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const themeSlider = document.getElementById('theme-slider');
    
    // Memory management
    const clearMemoryBtn = document.getElementById('clearMemory');
    const resetSystemBtn = document.getElementById('resetSystem');

    let isProcessing = false;

    // Add token usage display
    const tokenDisplay = document.createElement('div');
    tokenDisplay.className = 'token-display';
    document.querySelector('.header-content').appendChild(tokenDisplay);

    function updateTokenDisplay(warning = false) {
        if (warning) {
            tokenDisplay.classList.add('warning');
        } else {
            tokenDisplay.classList.remove('warning');
        }
    }

    function handleTokenWarning(message) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'message system-message warning';
        warningDiv.textContent = message;
        messagesContainer.appendChild(warningDiv);
        updateTokenDisplay(true);
        scrollToBottom();
    }

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

    // Custom message rendering based on type
    function createMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        if (isUser) {
            // User messages are always plain text
            messageDiv.textContent = content;
        } else {
            // Assistant messages can contain markdown/formatting
            messageDiv.innerHTML = marked.parse(content);
        }
        
        return messageDiv;
    }

    // Custom link rendering for maps
    renderer.link = function(href, title, text) {
        // Handle raw object input
        if (typeof href === 'object') {
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

        // For regular links, ensure we have descriptive text
        let displayText = text;
        if (!displayText || displayText === 'undefined' || displayText === '[object Object]' || displayText === href || displayText === 'Visit Website') {
            try {
                const url = new URL(href);
                // Extract domain without TLD
                const domain = url.hostname.split('.').slice(-2, -1)[0];
                
                if (domain) {
                    // Capitalize first letter of each word and remove special chars
                    const siteName = domain.split(/[-_]/)
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                        .join(' ');
                    
                    // Check URL path for common patterns
                    const path = url.pathname.toLowerCase();
                    if (path.includes('review')) {
                        displayText = `${siteName} Reviews`;
                    } else if (path.includes('order') || path.includes('delivery') || path.includes('pickup')) {
                        displayText = `Order on ${siteName}`;
                    } else if (path.includes('menu')) {
                        displayText = `View Menu on ${siteName}`;
                    } else if (path.includes('reservation') || path.includes('book')) {
                        displayText = `Make Reservation on ${siteName}`;
                    } else {
                        displayText = title || `View on ${siteName}`;
                    }
                } else {
                    displayText = title || 'Official Website';
                }
            } catch (e) {
                displayText = title || 'Official Website';
            }
        }
        
        return `<a href="${href}" ${title ? `title="${title}"` : ''} target="_blank">${displayText}</a>`;
    };

    // Add renderer to marked options
    marked.setOptions({ renderer: renderer });

    // Buffer for accumulating message content
    let messageBuffer = '';
    let typingTimer;
    const TYPING_DELAY = 50; // ms delay before rendering math
    
    // LaTeX delimiters to check for incomplete expressions
    const LATEX_DELIMITERS = {
        inline: [
            { start: '$', end: '$' },
            { start: '\\(', end: '\\)' }
        ],
        display: [
            { start: '$$', end: '$$' },
            { start: '\\[', end: '\\]' }
        ]
    };

    // Unique identifier for math placeholders
    let mathCounter = 0;
    const mathExpressions = new Map();

    // Function to create a placeholder for math content
    function createMathPlaceholder() {
        const id = `math-placeholder-${mathCounter++}`;
        return { id, placeholder: `%%${id}%%` };
    }

    // Function to extract and store math expressions
    function extractMathExpressions(content) {
        // Quick check if there's any potential math content
        if (!content.includes('$') && !content.includes('\\')) {
            return content;
        }

        let modifiedContent = content;
        mathExpressions.clear();
        
        // Process all types of math delimiters
        for (const type of ['display', 'inline']) {
            for (const delimiter of LATEX_DELIMITERS[type]) {
                let searchStart = 0;
                while (true) {
                    // Find next opening delimiter
                    const startIdx = modifiedContent.indexOf(delimiter.start, searchStart);
                    if (startIdx === -1) break;
                    
                    // Find matching closing delimiter
                    const endIdx = modifiedContent.indexOf(delimiter.end, startIdx + delimiter.start.length);
                    if (endIdx === -1) {
                        searchStart = startIdx + delimiter.start.length;
                        continue;
                    }
                    
                    // Extract the complete math expression
                    const fullExpression = modifiedContent.slice(startIdx, endIdx + delimiter.end.length);
                    const { id, placeholder } = createMathPlaceholder();
                    
                    // Store the expression with its type
                    mathExpressions.set(id, {
                        type,
                        expression: fullExpression,
                        rendered: false
                    });
                    
                    // Replace with placeholder
                    modifiedContent = modifiedContent.slice(0, startIdx) + placeholder + modifiedContent.slice(endIdx + delimiter.end.length);
                    searchStart = startIdx + placeholder.length;
                }
            }
        }
        
        return modifiedContent;
    }

    // Function to render a single math expression
    async function renderMathExpression(id, element) {
        const mathInfo = mathExpressions.get(id);
        if (!mathInfo || mathInfo.rendered) return;

        const placeholder = element.querySelector(`[data-math-id="${id}"]`);
        if (!placeholder) return;

        try {
            // Create a temporary container with specific styling
            const tempDiv = document.createElement('div');
            tempDiv.style.position = 'static';  // Prevent absolute positioning issues
            tempDiv.style.display = 'inline-block';
            tempDiv.style.maxWidth = '100%';
            tempDiv.style.overflow = 'auto';
            tempDiv.textContent = mathInfo.expression;

            // Clear any existing content
            placeholder.innerHTML = '';
            placeholder.appendChild(tempDiv);

            await MathJax.typesetPromise([tempDiv]);
            mathInfo.rendered = true;

            // Ensure SVG elements are properly sized
            const svg = tempDiv.querySelector('svg');
            if (svg) {
                svg.style.maxWidth = '100%';
                svg.style.position = 'static';
            }
        } catch (err) {
            console.error('MathJax rendering failed:', err);
            placeholder.textContent = mathInfo.expression;
        }
    }

    // Function to process and render content with math
    async function processContent(content, element) {
        // Quick check if content has changed
        if (element.dataset.lastContent === content) {
            return;
        }
        
        // Mark as new message for scrolling logic
        element.dataset.isNew = 'true';
        element.dataset.lastContent = content;

        // Extract math expressions and replace with placeholders
        const processedContent = extractMathExpressions(content);
        
        // Parse markdown
        let htmlContent = marked.parse(processedContent);
        
        // Replace placeholders with span elements
        for (const [id] of mathExpressions) {
            const placeholder = `%%${id}%%`;
            htmlContent = htmlContent.replace(
                placeholder,
                `<span class="math-placeholder" data-math-id="${id}">Loading math...</span>`
            );
        }
        
        // Update DOM
        element.innerHTML = htmlContent;
        
        // Process math expressions in chunks to maintain responsiveness
        const chunkSize = 5;
        const mathIds = Array.from(mathExpressions.keys());
        
        for (let i = 0; i < mathIds.length; i += chunkSize) {
            const chunk = mathIds.slice(i, i + chunkSize);
            await Promise.all(chunk.map(id => renderMathExpression(id, element)));
            // Small delay between chunks to maintain UI responsiveness
            if (i + chunkSize < mathIds.length) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        // Remove new message flag after processing
        setTimeout(() => {
            element.dataset.isNew = 'false';
        }, 100);

        scrollToBottom();
    }

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

    let shouldAutoScroll = true;  // Track if we should auto-scroll

    // Listen for manual scrolling
    messagesContainer.addEventListener('scroll', function() {
        const bottomThreshold = 100;  // pixels from bottom
        const isNearBottom = messagesContainer.scrollHeight - (messagesContainer.scrollTop + messagesContainer.clientHeight) < bottomThreshold;
        
        // Update auto-scroll flag based on user's scroll position
        shouldAutoScroll = isNearBottom;
    });

    function scrollToBottom() {
        // Only auto-scroll if user is already near bottom or it's a new message
        if (shouldAutoScroll) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
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

    // Add batch processing variables
    const BATCH_UPDATE_INTERVAL = 50; // ms
    let updateTimeout = null;
    let pendingTokens = '';

    async function batchProcessContent(content, element) {
        // Clear any pending timeout
        if (updateTimeout) {
            clearTimeout(updateTimeout);
        }

        // Accumulate tokens
        pendingTokens += content;

        // Schedule an update
        updateTimeout = setTimeout(async () => {
            await processContent(pendingTokens, element);
            pendingTokens = '';
            scrollToBottom();
        }, BATCH_UPDATE_INTERVAL);
    }

    // Debounce function for scroll updates
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Throttle function for frequent updates
    function throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }

    const debouncedScroll = debounce((force = false) => {
        const lastMessage = messagesContainer.lastElementChild;
        if (!lastMessage) return;

        const containerHeight = messagesContainer.clientHeight;
        const scrollPosition = messagesContainer.scrollTop;
        const scrollHeight = messagesContainer.scrollHeight;
        
        // Only scroll if we're already near the bottom or forced
        const isNearBottom = scrollHeight - (scrollPosition + containerHeight) < 100;
        if (force || isNearBottom) {
            messagesContainer.scrollTop = scrollHeight;
        }
    }, 100);

    // Throttled version of processContent
    const throttledProcess = throttle(async (content, element) => {
        // Skip if content hasn't changed significantly (at least 10 new characters)
        const lastContent = element.dataset.lastContent || '';
        if (content.length - lastContent.length < 10) {
            return;
        }

        element.dataset.lastContent = content;

        // Extract math expressions and replace with placeholders
        const processedContent = extractMathExpressions(content);
        
        // Parse markdown
        let htmlContent = marked.parse(processedContent);
        
        // Replace placeholders with span elements
        for (const [id] of mathExpressions) {
            const placeholder = `%%${id}%%`;
            htmlContent = htmlContent.replace(
                placeholder,
                `<span class="math-placeholder" data-math-id="${id}">Loading math...</span>`
            );
        }
        
        // Update DOM
        element.innerHTML = htmlContent;
        
        // Process math expressions
        const mathIds = Array.from(mathExpressions.keys());
        await Promise.all(mathIds.map(id => renderMathExpression(id, element)));

        // Always scroll to bottom during streaming
        scrollToBottom();
    }, 100);  // Reduced throttle time to 100ms

    async function sendMessage() {
        if (isProcessing || !userInput.value.trim()) return;
        
        isProcessing = true;
        sendButton.disabled = true;
        userInput.disabled = true;
        
        // Always scroll for new user messages
        shouldAutoScroll = true;
        
        // Create and display user message - preserve formatting
        const userMessage = userInput.value;  // Don't trim to preserve whitespace
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'message user-message';
        userMessageDiv.textContent = userMessage;  // textContent preserves whitespace
        messagesContainer.appendChild(userMessageDiv);
        scrollToBottom();
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Create assistant message container
        const assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.className = 'message assistant-message';
        messagesContainer.appendChild(assistantMessageDiv);
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            messageBuffer = '';
            mathCounter = 0;
            
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
                            if (data.warning) {
                                handleTokenWarning(data.warning);
                                continue;
                            }
                            if (data.token) {
                                messageBuffer += data.token;
                                await throttledProcess(messageBuffer, assistantMessageDiv);
                            }
                        } catch (e) {
                            console.error('Error parsing chunk:', e);
                        }
                    }
                }
            }
            
            // Final render after stream completes
            await processContent(messageBuffer, assistantMessageDiv);
            // Force scroll on completion only if we're still auto-scrolling
            if (shouldAutoScroll) {
                scrollToBottom();
            }
            
        } catch (error) {
            console.error('Error:', error);
            assistantMessageDiv.textContent = `Error: ${error.message}`;
            if (shouldAutoScroll) {
                scrollToBottom();
            }
        } finally {
            isProcessing = false;
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
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