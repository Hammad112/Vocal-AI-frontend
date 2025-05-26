document.addEventListener('DOMContentLoaded', () => {
    // State management object
    const appState = {
        tts: { text: '', gender: 'male', audioUrl: '', status: '' },
        sfx: { query: '', taskId: '', audioUrl: '', status: '' },
        voiceChanger: { effect: 'robot', audioUrl: '', processedUrl: '', status: '' },
        stt: { transcription: '', status: '' },
        activeSection: 'text-to-speech', // Default to text-to-speech
        menuOpen: false,
        exploreAnimated: false,
        
    };

    // Load state from localStorage
    const loadState = () => {
        const savedState = localStorage.getItem('appState');
        if (savedState) {
            const parsedState = JSON.parse(savedState);
            // Only load state for properties we want to persist
            appState.tts = parsedState.tts || { text: '', gender: 'male', audioUrl: '', status: '' };
            appState.sfx = parsedState.sfx || true;
            appState.activeSection = parsedState.activeSection || 'text-to-speech';
            appState.menuOpen = parsedState.menuOpen || false;
            appState.exploreAnimated = parsedState.exploreAnimated || false;
            appState.voiceChanger = parsedState.voiceChanger || {
                inputText: '',
                selectedEffect: 'robot',
                audioUrl: '',
                status: ''
            };
        }
        restoreUI();
    };
    

    // Save state to localStorage, excluding voiceChanger and stt
    const saveState = () => {
        const stateToSave = {
            tts: appState.tts,
            sfx: appState.sfx,
            activeSection: appState.activeSection,
            menuOpen: appState.menuOpen,
            exploreAnimated: appState.exploreAnimated,
            voiceChanger: appState.voiceChanger
        };
        localStorage.setItem('appState', JSON.stringify(stateToSave));
    };

    // Restore UI based on state
    const restoreUI = () => {
        // Ensure text-to-speech is the default section if no saved section exists
        if (!appState.activeSection || !document.querySelector(`.nav-btn-voice[data-section="${appState.activeSection}"]`)) {
            appState.activeSection = 'text-to-speech';
        }

        // Remove active class from all buttons and sections
        document.querySelectorAll('.nav-btn-voice').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.section').forEach(section => section.classList.remove('active'));

        // Activate the saved or default section
        const sectionBtn = document.querySelector(`.nav-btn-voice[data-section="${appState.activeSection}"]`);
        const section = document.querySelector(`.section[data-section="${appState.activeSection}"]`);

        if (sectionBtn && section) {
            sectionBtn.classList.add('active');
            section.classList.add('active');
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
            section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        } else {
            console.warn(`Section or button for "${appState.activeSection}" not found. Falling back to text-to-speech.`);
            appState.activeSection = 'text-to-speech';
            const defaultBtn = document.querySelector('.nav-btn-voice[data-section="text-to-speech"]');
            const defaultSection = document.querySelector('.section[data-section="text-to-speech"]');
            if (defaultBtn && defaultSection) {
                defaultBtn.classList.add('active');
                defaultSection.classList.add('active');
                defaultSection.style.opacity = '1';
                defaultSection.style.transform = 'translateY(0)';
                defaultSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            } else {
                console.error('Text-to-speech section or button not found in DOM.');
            }
        }

        const navLinks = document.querySelector('.nav-links');
        const menuToggle = document.querySelector('.menu-toggle');
        if (appState.menuOpen) {
            navLinks.style.display = 'flex';
            navLinks.style.opacity = '1';
            navLinks.style.transform = 'translateY(0)';
            navLinks.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            menuToggle.classList.add('active');
        } else {
            navLinks.style.transform = 'translateY(-10px)';
            menuToggle.classList.remove('active');
        }

        if (appState.exploreAnimated) {
            const voiceFeatures = document.getElementById('voice-features');
            if (voiceFeatures) {
                voiceFeatures.classList.add('animate-section');
                setTimeout(() => voiceFeatures.classList.remove('animate-section'), 1000);
            }
        }

        if (appState.tts.text) {
            document.getElementById('tts-input').value = appState.tts.text;
            document.querySelector(`#tts-${appState.tts.gender}`).classList.add('active');
            if (appState.tts.audioUrl) {
                const audio = document.getElementById('tts-audio');
                audio.src = appState.tts.audioUrl;
                audio.style.display = 'block';
                document.getElementById('tts-download').disabled = false;
                document.getElementById('tts-play').disabled = false;
            }
            updateStatus('tts-status', appState.tts.status, appState.tts.status.includes('Error') ? 'error' : 'success');
        }

        if (appState.sfx.query) {
            document.getElementById('sfx-input').value = appState.sfx.query;
            if (appState.sfx.audioUrl) {
                const audio = document.getElementById('sfx-audio');
                audio.src = appState.sfx.audioUrl;
                audio.classList.remove('hidden');
                updateStatus('sfx-status', appState.sfx.status, appState.sfx.status.includes('Error') ? 'error' : 'success');
            }
            if (appState.sfx.taskId && appState.sfx.status === 'processing') {
                pollForAudio(appState.sfx.taskId);
            }
        }

         // Restore voiceChanger input text
        if (appState.voiceChanger.inputText) {
            const vcInput = document.getElementById('voice-changer-input');
            if (vcInput) vcInput.value = appState.voiceChanger.inputText;
        }

        // Restore selected effect button for voiceChanger
        if (appState.voiceChanger.selectedEffect) {
            document.querySelectorAll('.voice-changer-effect-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.effect === appState.voiceChanger.selectedEffect) {
                    btn.classList.add('active');
                }
            });
        }

        // Restore audio if available
        if (appState.voiceChanger.audioUrl) {
            const vcAudio = document.getElementById('voice-changer-audio');
            if (vcAudio) {
                vcAudio.src = appState.voiceChanger.audioUrl;
                vcAudio.style.display = 'block';
            }
            const vcDownloadBtn = document.getElementById('voice-changer-download');
            if (vcDownloadBtn) vcDownloadBtn.disabled = false;
            const vcPlayBtn = document.getElementById('voice-changer-play');
            if (vcPlayBtn) vcPlayBtn.disabled = false;
        }

        // Update status if any
        if (appState.voiceChanger.status) {
            updateStatus('voice-changer-status', appState.voiceChanger.status, appState.voiceChanger.status.includes('Error') ? 'error' : 'success');
        }



        // No restoration for voiceChanger and stt since we're not saving their state
    };

    // Update status message
    const updateStatus = (elementId, message, className) => {
        const status = document.getElementById(elementId);
        if (status && message) {
            status.innerText = message;
            status.classList.remove('error', 'success', 'processing');
            status.classList.add(className);
        }
    };

    // Prevent page reloads during any generation
    let isGenerating = false;
    const preventReload = (e) => {
        if (isGenerating) {
            console.warn('Attempted reload blocked during generation', {
                eventType: e.type,
                location: window.location.href,
                timestamp: new Date().toISOString()
            });
            e.preventDefault();
            e.returnValue = 'Generation in progress. Are you sure?';
            return e.returnValue;
        }
    };
    
    window.addEventListener('beforeunload', preventReload);
    window.addEventListener('unload', preventReload);

    // Ensure single event listener
    const ensureSingleListener = (element, event, handler) => {
        if (!element) return null;
        const clonedElement = element.cloneNode(true);
        element.replaceWith(clonedElement);
        const wrappedHandler = (e) => {
            console.log(`Event triggered: ${event} on ${clonedElement.id || clonedElement.className} (Classes: ${clonedElement.classList}, Tag: ${clonedElement.tagName})`);
            handler(e);
        };
        clonedElement.removeEventListener(event, wrappedHandler);
        clonedElement.addEventListener(event, wrappedHandler, { once: false });
        return clonedElement;
    };

    // Prevent all form submissions
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            console.log('Form submission prevented for:', form.outerHTML);
        });
    });

    // Load initial state
    loadState();

    // Mobile menu toggle with animation
    let menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    menuToggle = ensureSingleListener(menuToggle, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const isOpen = navLinks.style.display === 'flex';
        appState.menuOpen = !isOpen;
        navLinks.style.display = isOpen ? 'none' : 'flex';
        navLinks.style.opacity = isOpen ? '0' : '1';
        navLinks.style.transform = isOpen ? 'translateY(-10px)' : 'translateY(0)';
        navLinks.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        menuToggle.classList.toggle('active');
        saveState();
        animateButton(menuToggle);
    });

    // Navigate to playground.html for explore features button
    let exploreButton = document.getElementById('explore-features');
    exploreButton = ensureSingleListener(exploreButton, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Explore features clicked');
        const href = exploreButton.getAttribute('href') || '/templates/playground.html';
        animateButton(exploreButton);
        console.log('Navigating to:', href);
        if (href) {
            appState.exploreAnimated = true;
            saveState();
            window.location.href = href;
        }
    });

    // Text-to-Speech (TTS) handlers
    let ttsMale = document.getElementById("tts-male");
    ttsMale = ensureSingleListener(ttsMale, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById("tts-male").classList.add("active");
        document.getElementById("tts-female").classList.remove("active");
        appState.tts.gender = 'male';
        saveState();
        animateButton(document.getElementById("tts-male"));
    });

    let ttsFemale = document.getElementById("tts-female");
    ttsFemale = ensureSingleListener(ttsFemale, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById("tts-female").classList.add("active");
        document.getElementById("tts-male").classList.remove("active");
        appState.tts.gender = 'female';
        saveState();
        animateButton(document.getElementById("tts-female"));
    });

    let ttsSubmit = document.getElementById("tts-submit");
    ttsSubmit = ensureSingleListener(ttsSubmit, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const text = document.getElementById("tts-input").value;
        const gender = document.querySelector(".voice-btn.active").id === "tts-male" ? "male" : "female";
        const status = document.getElementById("tts-status");
        const submitBtn = document.getElementById("tts-submit");

        if (!text) {
            updateStatus('tts-status', 'Error: Please enter text.', 'error');
            appState.tts.status = 'Error: Please enter text.';
            saveState();
            return;
        }

        const formData = new FormData();
        formData.append("text", text);
        formData.append("gender", gender);

        try {
            isGenerating = true;
            updateStatus('tts-status', 'Generating...', 'processing');
            appState.tts.status = 'Generating...';
            appState.tts.text = text;
            appState.tts.gender = gender;
            saveState();
            if (submitBtn) submitBtn.classList.add('loading');

            const response = await fetch("http://localhost:8000/generate", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            appState.tts.audioUrl = `http://localhost:8000/download?t=${new Date().getTime()}`;
            appState.tts.status = data.message;
            saveState();

            updateStatus('tts-status', data.message, 'success');
            document.getElementById("tts-download").disabled = false;
            document.getElementById("tts-play").disabled = false;

            const audio = document.getElementById("tts-audio");
            if (audio) {
                audio.src = appState.tts.audioUrl;
                audio.style.display = "block";
                audio.load();
                audio.play().catch(err => {
                    updateStatus('tts-status', `Error playing audio: ${err.message}`, 'error');
                    appState.tts.status = `Error playing audio: ${err.message}`;
                    saveState();
                });
                animateAudioPlayback(audio);
            }
        } catch (error) {
            updateStatus('tts-status', `Error: ${error.message}`, 'error');
            appState.tts.status = `Error: ${error.message}`;
            saveState();
        } finally {
            isGenerating = false;
            if (submitBtn) submitBtn.classList.remove('loading');
        }
    });

    let ttsPlay = document.getElementById("tts-play");
    ttsPlay = ensureSingleListener(ttsPlay, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const audio = document.getElementById("tts-audio");
        const audioUrl = `http://localhost:8000/download?t=${new Date().getTime()}`;

        try {
            const response = await fetch(audioUrl);
            if (!response.ok) {
                throw new Error("Audio file not found.");
            }

            if (audio) {
                audio.src = audioUrl;
                appState.tts.audioUrl = audioUrl;
                saveState();
                audio.style.display = "block";
                audio.load();
                audio.play().catch(err => {
                    updateStatus('tts-status', `Error playing audio: ${err.message}`, 'error');
                    appState.tts.status = `Error playing audio: ${err.message}`;
                    saveState();
                });
                animateAudioPlayback(audio);
            }
        } catch (error) {
            updateStatus('tts-status', `Error: ${error.message}`, 'error');
            appState.tts.status = `Error: ${error.message}`;
            saveState();
        }
    });

    let ttsDownload = document.getElementById("tts-download");
    ttsDownload = ensureSingleListener(ttsDownload, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const downloadUrl = `http://localhost:8000/download?t=${new Date().getTime()}`;
        const status = document.getElementById("tts-status");

        try {
            const response = await fetch(downloadUrl);
            if (!response.ok) {
                throw new Error("Download file not available.");
            }

            const link = document.createElement("a");
            link.href = downloadUrl;
            link.download = "speech.mp3";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            animateButton(document.getElementById("tts-download"));
        } catch (error) {
            updateStatus('tts-status', `Error: ${error.message}`, 'error');
            appState.tts.status = `Error: ${error.message}`;
            saveState();
        }
    });

    // Speech-to-Text (STT) handlers
    let mediaRecorder;
    let audioChunks = [];
    let startButton = document.getElementById('start-recording');
    const output = document.getElementById('stt-output');

    if (!navigator.mediaDevices || !window.MediaRecorder) {
        if (output) output.textContent = 'MediaRecorder not supported in this browser.';
        if (startButton) startButton.disabled = true;
    } else {
        startButton = ensureSingleListener(startButton, 'click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (startButton.textContent === 'Start Recording') {
                try {
                    isGenerating = true;
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        audioChunks = [];
                        const formData = new FormData();
                        formData.append('file', audioBlob, 'recording.wav');

                        updateStatus('stt-output', 'Transcribing...', 'processing');
                        appState.stt.status = 'Transcribing...';

                        try {
                            const response = await fetch('http://localhost:8000/transcribe', {
                                method: 'POST',
                                body: formData,
                            });

                            if (!response.ok) {
                                throw new Error(`Transcription failed: ${response.statusText}`);
                            }

                            const result = await response.json();
                            appState.stt.transcription = result.transcription || 'No transcription available';
                            appState.stt.status = 'Transcription completed';
                            updateStatus('stt-output', appState.stt.transcription, 'success');
                        } catch (error) {
                            updateStatus('stt-output', `Error: ${error.message}`, 'error');
                            appState.stt.status = `Error: ${error.message}`;
                        }

                        stream.getTracks().forEach(track => track.stop());
                        isGenerating = false;
                    };

                    mediaRecorder.start();
                    startButton.textContent = 'Stop Recording';
                    startButton.classList.add('recording');
                    animateButton(startButton);
                } catch (error) {
                    updateStatus('stt-output', `Error accessing microphone: ${error.message}`, 'error');
                    appState.stt.status = `Error accessing microphone: ${error.message}`;
                    isGenerating = false;
                }
            } else {
                mediaRecorder.stop();
                startButton.textContent = 'Start Recording';
                startButton.classList.remove('recording');
                animateButton(startButton);
            }
        });
    }

    // Voice Changer handlers
    let recordVoice = document.getElementById("record-voice");
    recordVoice = ensureSingleListener(recordVoice, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
            isGenerating = true;
            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.start();
            updateStatus('recording-status', 'Recording...', 'processing');
            appState.voiceChanger.status = 'Recording...';

            recordVoice.classList.add("hidden");
            document.getElementById("stop-recording").classList.remove("hidden");
            animateButton(recordVoice);

            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                let audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                let audioUrl = URL.createObjectURL(audioBlob);
                appState.voiceChanger.audioUrl = audioUrl;

                let recordedAudio = document.getElementById("recorded-audio");
                if (recordedAudio) {
                    recordedAudio.src = audioUrl;
                    recordedAudio.classList.remove("hidden");
                }
                document.getElementById("effect-controls").classList.remove("hidden");
                document.getElementById("apply-voice").classList.remove("hidden");
                updateStatus('recording-status', 'Recording stopped.', 'success');
                appState.voiceChanger.status = 'Recording stopped.';
                if (recordedAudio) animateAudioPlayback(recordedAudio);
                isGenerating = false;
            };
        } catch (err) {
            updateStatus('recording-status', `Error accessing microphone: ${err.message}`, 'error');
            appState.voiceChanger.status = `Error accessing microphone: ${err.message}`;
            isGenerating = false;
        }
    });

    let stopRecording = document.getElementById("stop-recording");
    stopRecording = ensureSingleListener(stopRecording, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (mediaRecorder) mediaRecorder.stop();
        stopRecording.classList.add("hidden");
        document.getElementById("record-voice").classList.remove("hidden");
        animateButton(stopRecording);
    });

    let applyVoice = document.getElementById("apply-voice");
    applyVoice = ensureSingleListener(applyVoice, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        let effect = document.getElementById("voice-effect").value;
        let recordedAudio = document.getElementById("recorded-audio");

        if (!recordedAudio || !recordedAudio.src) {
            updateStatus('recording-status', 'Error: No audio recorded.', 'error');
            appState.voiceChanger.status = 'Error: No audio recorded.';
            return;
        }

        let audioBlob = await fetch(recordedAudio.src).then(res => res.blob());
        let formData = new FormData();
        formData.append("audio", audioBlob, "audio.wav");
        formData.append("effect", effect);

        try {
            isGenerating = true;
            updateStatus('recording-status', 'Processing...', 'processing');
            appState.voiceChanger.status = 'Processing...';
            appState.voiceChanger.effect = effect;

            applyVoice.classList.add('loading');

            let response = await fetch("http://localhost:8000/process", {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                let processedBlob = await response.blob();
                let processedUrl = URL.createObjectURL(processedBlob);
                appState.voiceChanger.processedUrl = processedUrl;

                let transformedAudio = document.getElementById("voice-changer-audio");
                if (transformedAudio) {
                    transformedAudio.src = processedUrl;
                    transformedAudio.classList.remove("hidden");
                    await new Promise((resolve, reject) => {
                        transformedAudio.addEventListener('loadedmetadata', resolve, { once: true });
                        transformedAudio.addEventListener('error', () => reject(new Error('Failed to load audio metadata')), { once: true });
                        setTimeout(() => reject(new Error('Timeout waiting for loadedmetadata')), 5000);
                    });

                    transformedAudio.play().then(() => {
                        animateAudioPlayback(transformedAudio);
                    }).catch(err => {
                        updateStatus('recording-status', `Error playing audio: ${err.message}`, 'error');
                        appState.voiceChanger.status = `Error playing audio: ${err.message}`;
                    });

                    let downloadBtn = document.getElementById("download-voice");
                    if (downloadBtn) {
                        downloadBtn.classList.remove("hidden");
                        downloadBtn.dataset.url = processedUrl;
                    }
                    updateStatus('recording-status', 'Audio processed successfully.', 'success');
                    appState.voiceChanger.status = 'Audio processed successfully.';
                }
            } else {
                const errorText = await response.text();
                throw new Error(`Error processing audio: ${response.statusText} - ${errorText}`);
            }
        } catch (err) {
            updateStatus('recording-status', `Error: ${err.message}`, 'error');
            appState.voiceChanger.status = `Error: ${err.message}`;
        } finally {
            isGenerating = false;
            applyVoice.classList.remove('loading');
        }
    });

    let downloadVoice = document.getElementById("download-voice");
    downloadVoice = ensureSingleListener(downloadVoice, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const url = document.getElementById("download-voice").dataset.url;
        const effect = document.getElementById("voice-effect").value;
        if (url) {
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            a.download = `transformed-voice-${effect}.wav`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            animateButton(downloadVoice);
        }
    });

    let sfxSubmit = document.getElementById("sfx-submit");
    sfxSubmit = ensureSingleListener(sfxSubmit, 'click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const query = document.getElementById("sfx-input").value.trim();
        const sfxAudio = document.getElementById("sfx-audio");
        let status = document.getElementById("sfx-status");
        if (!status) {
            status = document.createElement('div');
            status.id = 'sfx-status';
            status.className = 'status-message';
            if (sfxSubmit && sfxSubmit.parentElement) sfxSubmit.parentElement.appendChild(status);
        }

        if (!query) {
            updateStatus('sfx-status', 'Error: Please enter text for SFX generation.', 'error');
            appState.sfx.status = 'Error: Please enter text for SFX generation.';
            saveState();
            return;
        }

        const pollForAudio = async (taskId) => {
            const maxPolls = 60;
            let pollCount = 0;
            const maxRetries = 5;
            const baseDelay = 1000;

            while (pollCount < maxPolls) {
                let retries = 0;
                while (retries < maxRetries) {
                    try {
                        console.log(`Polling attempt ${pollCount + 1}/${maxPolls}, retry ${retries + 1}/${maxRetries} for task ${taskId}`);
                        const response = await fetch(`http://localhost:8000/check_sfx_status/${taskId}`, {
                            method: 'GET',
                            headers: { 'Cache-Control': 'no-cache' }
                        });
                        if (!response.ok) {
                            throw new Error(`Status check failed: ${response.statusText}`);
                        }
                        const data = await response.json();
                        if (data.status === 'completed') {
                            const audioUrl = `http://localhost:8000${data.url}?v=${Date.now()}`;
                            appState.sfx.audioUrl = audioUrl;
                            appState.sfx.status = 'SFX generated and playing.';
                            saveState();
                            if (sfxAudio) {
                                sfxAudio.src = audioUrl;
                                sfxAudio.classList.remove("hidden");
                                sfxAudio.addEventListener("canplaythrough", () => {
                                    sfxAudio.play().catch(err => {
                                        updateStatus('sfx-status', `Error playing SFX: ${err.message}`, 'error');
                                        appState.sfx.status = `Error playing SFX: ${err.message}`;
                                        saveState();
                                    });
                                    animateAudioPlayback(sfxAudio);
                                }, { once: true });
                            }
                            updateStatus('sfx-status', 'SFX generated and playing.', 'success');
                            return;
                        } else if (data.status === 'failed') {
                            throw new Error(data.error || 'SFX generation failed');
                        }
                        updateStatus('sfx-status', `Generating SFX... (${pollCount + 1}/${maxPolls})`, 'processing');
                        appState.sfx.status = `Generating SFX... (${pollCount + 1}/${maxPolls})`;
                        saveState();
                        pollCount++;
                        break;
                    } catch (error) {
                        retries++;
                        console.warn(`Poll attempt ${pollCount + 1}, retry ${retries} failed: ${error.message}`);
                        if (retries >= maxRetries) {
                            updateStatus('sfx-status', `Error: ${error.message}`, 'error');
                            appState.sfx.status = `Error: ${error.message}`;
                            saveState();
                            return;
                        }
                        const delay = baseDelay * Math.pow(2, retries);
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            updateStatus('sfx-status', 'Error: SFX generation timed out.', 'error');
            appState.sfx.status = 'Error: SFX generation timed out.';
            saveState();
        };

        try {
            isGenerating = true;
            updateStatus('sfx-status', 'Starting SFX generation...', 'processing');
            appState.sfx.status = 'Starting SFX generation...';
            appState.sfx.query = query;
            saveState();
            sfxSubmit.disabled = true;
            sfxSubmit.textContent = "Generating...";
            sfxSubmit.classList.add('loading');

            const response = await fetch("http://localhost:8000/generate_sfx", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: query }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || `Failed to generate SFX: ${response.statusText}`);
            }

            const data = await response.json();
            appState.sfx.taskId = data.task_id;
            appState.sfx.status = 'processing';
            saveState();
            console.log(`Starting polling for task: ${data.task_id}`);
            await pollForAudio(data.task_id);
        } catch (error) {
            updateStatus('sfx-status', `Error: ${error.message}`, 'error');
            appState.sfx.status = `Error: ${error.message}`;
            saveState();
        } finally {
            isGenerating = false;
            sfxSubmit.disabled = false;
            sfxSubmit.textContent = "Generate SFX";
            sfxSubmit.classList.remove('loading');
        }
    });

    // Voice section navigation
    document.querySelectorAll('.nav-btn-voice').forEach(btn => {
        ensureSingleListener(btn, 'click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Handle create-clone-btn navigation
            if (btn.id === 'create-clone-btn') {
                const link = btn.querySelector('a');
                const href = link ? link.getAttribute('href') : '/templates/playground.html';
                console.log('Create clone clicked, navigating to:', href);
                animateButton(btn, true);
                if (href) {
                    appState.activeSection = 'create-clone';
                    saveState();
                    window.location.href = href;
                }
                return;
            }

            // Handle other nav-btn-voice buttons
            document.querySelector('.nav-btn-voice.active')?.classList.remove('active');
            document.querySelector('.section.active')?.classList.remove('active');

            btn.classList.add('active');
            const sectionId = btn.getAttribute('data-section');
            appState.activeSection = sectionId;
            saveState();

            const section = document.querySelector(`.section[data-section="${sectionId}"]`);
            if (section) {
                section.classList.add('active');
                section.style.opacity = '0';
                section.style.transform = 'translateY(20px)';
                section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                setTimeout(() => {
                    section.style.opacity = '1';
                    section.style.transform = 'translateY(0)';
                }, 50);
            }

            animateButton(btn);
        });
    });

    // Workflow diagram node animations
    const nodes = document.querySelectorAll('.node');
    nodes.forEach(node => {
        node.addEventListener('mouseenter', () => {
            node.style.transform = 'translate(-50%, -60%) scale(1.1)';
            node.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.3)';
            node.style.transition = 'all 0.3s ease';
            node.classList.add('pulse');
        });
        node.addEventListener('mouseleave', () => {
            node.style.transform = 'translate(-50%, -50%) scale(1)';
            node.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
            node.classList.remove('pulse');
        });
    });

    // Voice cloning play buttons
    document.querySelectorAll('.play-btn').forEach(btn => {
        ensureSingleListener(btn, 'click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Voice cloning play button clicked');
            const profile = btn.closest('.profile')?.querySelector('span')?.textContent.toLowerCase();
            const variant = btn.closest('.profile')?.classList.contains('original') ? 'original' : 'clone';
            if (!profile) return;

            const audio = document.createElement('audio');

            try {
                const audioUrl = `http://localhost:8000/voices/${profile}/${variant}?t=${new Date().getTime()}`;
                audio.src = audioUrl;
                await audio.load();
                audio.play().catch(err => {
                    console.error(`Error playing ${profile} ${variant}:`, err);
                    alert(`Error playing voice: ${err.message}`);
                });
                animateButton(btn);
            } catch (error) {
                console.error(`Error loading voice for ${profile} ${variant}:`, error);
                alert(`Error: ${error.message}`);
            }
        });
    });

    function animateButton(button, isCreateClone = false) {
        if (button) {
            button.classList.add('button-click');
            if (isCreateClone) {
                button.style.background = '#e6b800';
                setTimeout(() => {
                    button.style.background = '';
                }, 300);
            }
            setTimeout(() => button.classList.remove('button-click'), 300);
        }
    }

    function animateAudioPlayback(audioElement) {
        if (audioElement && audioElement.parentElement) {
            const parent = audioElement.parentElement;
            parent.classList.add('audio-playing');
            audioElement.addEventListener('ended', () => {
                parent.classList.remove('audio-playing');
            }, { once: true });
        }
    }
});
