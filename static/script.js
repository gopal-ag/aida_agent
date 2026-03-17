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

    function formatMarkdown(text) {
        // Simple markdown parsing for the demo
        
        // 1. Format code blocks
        let parsed = text.replace(/```([\s\S]*?)```/g, function(match, code) {
            // Strip language identifier if present on first line
            let codeContent = code.replace(/^[a-z]+\n/, ''); 
            return `<pre><code>${codeContent}</code></pre>`;
        });

        // 2. Format CSV-like table for the exact diff response
        if (parsed.includes("acc_no,swift_score,open_score,diff_open_minus_swift")) {
            const tableRows = `
                <table class="data-table">
                    <thead>
                        <tr><th>acc_no</th><th>swift_score</th><th>open_score</th><th>diff_open_minus_swift</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>37.454011885</td><td>717.882</td><td>713.379</td><td>-4.502999999999929</td></tr>
                        <tr><td>50.313625858</td><td>471.684</td><td>476.006</td><td>4.321999999999946</td></tr>
                        <tr><td>53.258943255</td><td>436.111</td><td>435.039</td><td>-1.0720000000000027</td></tr>
                    </tbody>
                </table>
            `;
            // Replace the hardcoded raw text with the table representation
            parsed = parsed.replace(/Example deltas[\s\S]*?(?=\n\nHere is)/, `Example deltas:\n${tableRows}`);
        }

        // 3. Format download links
        parsed = parsed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" download class="download-link" style="color: #646cff; text-decoration: underline;">$1</a>');
        
        // 4. Line breaks
        parsed = parsed.replace(/\n/g, '<br>');
        
        return parsed;
    }

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        contentDiv.innerHTML = formatMarkdown(text);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendLoading(text = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message assistant loading-msg`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content loading';
        
        let innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        if (text) {
             innerHTML += `<div style="margin-top: 10px; font-size: 0.9em; color: var(--text-secondary);">${text}</div>`;
        }
        contentDiv.innerHTML = innerHTML;
        
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
    
    // Simulate typical LLM typing/thinking delay (10s)
    function getThinkingDelay() {
        return 10000;
    }
    
    // Simulate script execution delay (~20s)
    function getExecutionDelay() {
        return Math.floor(Math.random() * 2000) + 19000;
    }

    async function sendMessage(overrideText = null, isApproval = false) {
        const text = overrideText || chatInput.value.trim();
        if (!text) return;

        // Hide panels on new message
        uploadPanel.classList.add('hidden');
        approvalPanel.classList.add('hidden');

        appendMessage('user', text);
        if (!overrideText) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
            sendBtn.setAttribute('disabled', 'true');
        }

        if (isApproval) {
            appendLoading("Executing script in sandbox environment...");
        } else {
            appendLoading("AiDa is thinking...");
        }

        try {
            // Note: If it's an approval, we just send it as a regular chat message 
            // since the agent state machine advances when "yes" or "approve" is in chat.
            // Sending to /approve was interrupting the linear chat flow causing duplicates.
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, thread_id: threadId })
            });
            const data = await response.json();
            
            // Artificial delay for realism
            const delay = isApproval ? getExecutionDelay() : getThinkingDelay();
            setTimeout(() => {
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
            }, delay);

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
                    sendMessage("I've uploaded the requested artifacts. Please run your diagnostic tools now.", false);
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
    approveBtn.addEventListener('click', () => {
        sendMessage("Yes, approved.", true);
    });
});
