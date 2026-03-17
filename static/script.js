document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const uploadPanel = document.getElementById('upload-panel');
    const approvalPanel = document.getElementById('approval-panel');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const cancelUploadBtn = document.getElementById('cancel-upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const approveBtn = document.getElementById('approve-btn');

    // Thread ID generation
    const threadId = 'thread_' + Math.random().toString(36).substr(2, 9);
    
    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value.trim().length > 0) {
            sendBtn.removeAttribute('disabled');
        } else {
            sendBtn.setAttribute('disabled', 'true');
        }
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (chatInput.value.trim().length > 0) {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        let parsedText = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" download class="download-link" style="color: #646cff; text-decoration: underline;">$1</a>');
        parsedText = parsedText.replace(/\n/g, '<br>');
        
        contentDiv.innerHTML = parsedText;
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendLoading() {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message assistant loading-msg`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content loading';
        contentDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function removeLoading() {
        const loadingMsgs = document.querySelectorAll('.loading-msg');
        loadingMsgs.forEach(m => m.remove());
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Hide panels on new message
        uploadPanel.classList.add('hidden');
        approvalPanel.classList.add('hidden');

        appendMessage('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.setAttribute('disabled', 'true');

        const loadingMarker = appendLoading();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, thread_id: threadId })
            });
            const data = await response.json();
            
            removeLoading();
            appendMessage('assistant', data.response);

            if (data.upload_required) {
                uploadPanel.classList.remove('hidden');
                uploadStatus.textContent = '';
                fileInput.value = '';
            }

            if (data.requires_approval) {
                approvalPanel.classList.remove('hidden');
            }

        } catch (error) {
            removeLoading();
            appendMessage('assistant', 'Error connecting to the agent backend.');
        }
    }

    // Upload functionality
    cancelUploadBtn.addEventListener('click', () => {
        uploadPanel.classList.add('hidden');
    });

    uploadBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) {
            uploadStatus.textContent = 'Please select files first.';
            uploadStatus.style.color = 'var(--warning-color)';
            return;
        }

        uploadStatus.textContent = 'Uploading...';
        uploadStatus.style.color = 'var(--text-secondary)';
        
        const formData = new FormData();
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append('files', fileInput.files[i]);
        }

        try {
            const res = await fetch(`/upload?thread_id=${threadId}`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                uploadStatus.textContent = 'Upload successful!';
                uploadStatus.style.color = 'var(--success-color)';
                setTimeout(() => {
                    uploadPanel.classList.add('hidden');
                    // Auto trigger chat so agent processes artifacts
                    chatInput.value = "I've uploaded the requested artifacts. Please run your diagnostic tools now.";
                    sendMessage();
                }, 1000);
            } else {
                uploadStatus.textContent = 'Upload failed.';
                uploadStatus.style.color = 'var(--warning-color)';
            }
        } catch (e) {
            uploadStatus.textContent = 'Upload error.';
            uploadStatus.style.color = 'var(--warning-color)';
        }
    });

    // Approval functionality
    approveBtn.addEventListener('click', async () => {
        approvalPanel.classList.add('hidden');
        appendMessage('user', 'Yes, approved.');
        const loadingMarker = appendLoading();

        try {
            const response = await fetch('/approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: threadId })
            });
            const data = await response.json();
            
            removeLoading();
            appendMessage('assistant', data.response);
        } catch (error) {
            removeLoading();
            appendMessage('assistant', 'Error handling approval.');
        }
    });
});
