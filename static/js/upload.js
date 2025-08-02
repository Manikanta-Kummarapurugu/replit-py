// Upload form handling and progress tracking
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const videoInput = document.getElementById('video');

    // File validation
    videoInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            // Check file size (500MB limit)
            const maxSize = 500 * 1024 * 1024; // 500MB in bytes
            if (file.size > maxSize) {
                alert('File size exceeds 500MB limit. Please select a smaller file.');
                this.value = '';
                return;
            }

            // Check file type
            const allowedTypes = [
                'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
                'video/x-ms-wmv', 'video/x-flv', 'video/webm', 'video/x-matroska'
            ];
            
            if (!allowedTypes.includes(file.type)) {
                alert('Invalid file type. Please select a valid video file.');
                this.value = '';
                return;
            }

            // Update button text with file info
            updateButtonText(file);
        }
    });

    // Form submission handling
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!videoInput.files[0]) {
            alert('Please select a video file to upload.');
            return;
        }

        // Show upload progress
        showUploadProgress();
        
        // Submit form
        this.submit();
    });

    function updateButtonText(file) {
        const fileName = file.name.length > 30 ? 
            file.name.substring(0, 30) + '...' : file.name;
        const fileSize = formatFileSize(file.size);
        
        uploadBtn.innerHTML = `
            <i class="fas fa-upload me-2"></i>
            Upload "${fileName}" (${fileSize})
        `;
    }

    function showUploadProgress() {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `
            <i class="fas fa-spinner fa-spin me-2"></i>
            Uploading...
        `;
        uploadProgress.style.display = 'block';
        
        // Animate progress bar
        const progressBar = uploadProgress.querySelector('.progress-bar');
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 90) {
                clearInterval(interval);
                progressBar.innerHTML = 'Processing...';
            } else {
                width += Math.random() * 10;
                progressBar.style.width = Math.min(width, 90) + '%';
            }
        }, 500);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Drag and drop functionality
    const formBody = uploadForm.querySelector('.card-body');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        formBody.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        formBody.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        formBody.addEventListener(eventName, unhighlight, false);
    });

    formBody.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        formBody.classList.add('border-primary');
        formBody.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
    }

    function unhighlight() {
        formBody.classList.remove('border-primary');
        formBody.style.backgroundColor = '';
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            videoInput.files = files;
            videoInput.dispatchEvent(new Event('change'));
        }
    }

    // Auto-refresh status for processing videos
    function checkProcessingStatus() {
        const statusElements = document.querySelectorAll('[data-video-id]');
        
        statusElements.forEach(element => {
            const videoId = element.getAttribute('data-video-id');
            const currentStatus = element.getAttribute('data-status');
            
            if (currentStatus === 'processing') {
                fetch(`/api/video/${videoId}/status`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status !== 'processing') {
                            // Reload page if status changed
                            location.reload();
                        }
                    })
                    .catch(error => {
                        console.log('Status check failed:', error);
                    });
            }
        });
    }

    // Check status every 5 seconds if there are processing videos
    if (document.querySelector('[data-status="processing"]')) {
        setInterval(checkProcessingStatus, 5000);
    }

    // Add tooltips to help explain features
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + U for upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            videoInput.click();
        }
        
        // ESC to cancel/clear
        if (e.key === 'Escape') {
            if (videoInput.value) {
                videoInput.value = '';
                uploadBtn.innerHTML = `
                    <i class="fas fa-upload me-2"></i>
                    Upload and Analyze Video
                `;
            }
        }
    });

    // Add visual feedback for form validation
    videoInput.addEventListener('invalid', function() {
        this.classList.add('is-invalid');
    });

    videoInput.addEventListener('input', function() {
        this.classList.remove('is-invalid');
    });

    // Performance monitoring
    const startTime = performance.now();
    window.addEventListener('load', function() {
        const loadTime = performance.now() - startTime;
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    });
});

// Global error handler for fetch requests
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});

// Service worker registration for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Note: Service worker file would need to be created separately
        // navigator.serviceWorker.register('/sw.js');
    });
}
