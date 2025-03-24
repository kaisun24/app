document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI elements
    initializeUI();
    
    // Setup event listeners
    setupEventListeners();
});

function initializeUI() {
    // Setup file upload
    setupFileUpload();
    
    // Setup tabs
    setupTabs();
    
    // Initialize chat
    setupChat();
}

function setupFileUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');

    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
        dropzone.style.backgroundColor = 'rgba(66, 133, 244, 0.05)';
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        dropzone.style.backgroundColor = 'transparent';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    analyzeBtn.addEventListener('click', () => {
        if (fileInput.files.length) {
            analyzeCode(fileInput.files[0]);
        }
    });
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const navLinks = document.querySelectorAll('.nav-link');
    const tabPanes = document.querySelectorAll('.tab-pane');

    function activateTab(tabId) {
        // Deactivate all tabs and panes
        tabs.forEach(tab => tab.classList.remove('active'));
        navLinks.forEach(link => link.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));

        // Activate selected tab and pane
        document.querySelector(`.tab[data-content="${tabId}"]`).classList.add('active');
        document.querySelector(`.nav-link[data-tab="${tabId}"]`).classList.add('active');
        document.getElementById(tabId).classList.add('active');
    }

    // Add click handlers
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            activateTab(tab.dataset.content);
        });
    });

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            activateTab(link.dataset.tab);
        });
    });
}

function setupChat() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');
    const clearButton = document.getElementById('clear-chat');
    const chatMessages = document.getElementById('chat-messages');

    sendButton.addEventListener('click', () => {
        sendMessage();
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    clearButton.addEventListener('click', () => {
        clearChat();
    });

    function sendMessage() {
        const message = chatInput.value.trim();
        if (message) {
            addMessage('user', message);
            chatInput.value = '';
            // Simulate response (replace with actual API call)
            setTimeout(() => {
                handleCommand(message);
            }, 500);
        }
    }

    function addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function clearChat() {
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
    }
}

function handleCommand(message) {
    if (message.startsWith('/')) {
        const command = message.split(' ')[0].toLowerCase();
        const args = message.slice(command.length).trim();

        switch (command) {
            case '/github':
                searchGitHub(args);
                break;
            case '/stack':
                searchStackOverflow(args);
                break;
            case '/explain':
                explainCode(args);
                break;
            default:
                addBotMessage("Unknown command. Try /github, /stack, or /explain");
        }
    } else {
        // Regular message handling
        simulateResponse(message);
    }
}

function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        analyzeCode(content);
    };
    reader.readAsText(file);
}

function analyzeCode(content) {
    // Simulate code analysis (replace with actual API call)
    const metrics = {
        language: "Python",
        totalLines: 100,
        codeLines: 80,
        comments: 20,
        complexity: 5
    };

    updateMetrics(metrics);
    generateRecommendations(metrics);
}

function updateMetrics(metrics) {
    const metricsDiv = document.getElementById('metrics');
    metricsDiv.innerHTML = `
        <h3>Code Metrics</h3>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-value">${metrics.language}</div>
                <div class="metric-label">Language</div>
            </div>
            <div class="metric">
                <div class="metric-value">${metrics.totalLines}</div>
                <div class="metric-label">Total Lines</div>
            </div>
            <!-- Add more metrics as needed -->
        </div>
    `;
}

function generateRecommendations(metrics) {
    const recommendationsDiv = document.getElementById('recommendations');
    // Generate recommendations based on metrics
    const recommendations = [
        "Consider adding more comments to improve code readability",
        "Break down complex functions to reduce cyclomatic complexity",
        "Add error handling for robust code"
    ];

    recommendationsDiv.innerHTML = `
        <h3>Recommendations</h3>
        <ul>
            ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
    `;
}

// API call simulations (replace with actual API calls)
function searchGitHub(query) {
    addBotMessage(`Searching GitHub for: ${query}`);
    // Implement actual GitHub API call
}

function searchStackOverflow(query) {
    addBotMessage(`Searching Stack Overflow for: ${query}`);
    // Implement actual Stack Overflow API call
}

function explainCode(code) {
    addBotMessage("Analyzing code...");
    // Implement actual code explanation logic
}

function simulateResponse(message) {
    addBotMessage(`I received your message: ${message}`);
}

function addBotMessage(content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
} 