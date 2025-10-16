import time
from collections import deque

from vita_e.vita_vla_html.web.queue import PCMQueue, ThreadSafeQueue
from vita_e.wakeup_and_vad.wakeup_and_vad import WakeupAndVAD

class GlobalParams:
    """
    Global parameters for managing conversation state, TTS, and video frame collection.
    Handles wakeup detection, voice activity detection, and multimodal data buffering.
    """
    def __init__(self):
        """Initialize global parameters with default values and reset state."""
        # Initialize wakeup and voice activity detection module
        self.wakeup_and_vad = WakeupAndVAD("./vita_e/wakeup_and_vad/resource", cache_history=10)
        self.interrupt_signal = 1
        self.collected_images = deque(maxlen=8)  # 存储最近8帧图像 (Store recent 8 frames)
        self.last_image_time = time.time()  # 记录最后一帧图像的时间 (Record timestamp of last frame)
        print("GlobalParams init")
        self.reset()

    def reset(self):
        """Reset all state variables to initial values."""
        # Generation control flags
        self.stop_generate = False
        self.is_generate = False
        self.wakeup_and_vad.in_dialog = False
        self.whole_text = ""

        # TTS (Text-to-Speech) control variables
        self.tts_over = False
        self.tts_over_time = 0
        self.tts_data = ThreadSafeQueue()
        self.pcm_fifo_queue = PCMQueue()

        # Stop flags for TTS and PCM playback
        self.stop_tts = False
        self.stop_pcm = False

        # Clear collected video frames
        self.collected_images.clear()
        self.last_image_time = time.time()
    
    def interrupt(self):
        """
        Interrupt the current generation and TTS playback.
        Waits for generation to stop, then clears TTS queue and video frames.
        """
        # Signal to stop generation and TTS
        self.stop_generate = True
        self.tts_over = True
        
        # Wait for generation to complete
        while True:
            time.sleep(0.01)
            if not self.is_generate:
                self.stop_generate = False
                # Wait for TTS queue to be empty
                while True:
                    time.sleep(0.01)
                    if self.tts_data.is_empty():
                        self.whole_text = ""
                        self.tts_over = False
                        self.tts_over_time += 1
                        # 清空视频帧 (Clear video frames)
                        self.collected_images.clear()
                        break
                break
    
    def release(self):
        """
        Release resources and clear video frame buffer.
        """
        # 清空视频帧 (Clear video frames)
        self.collected_images.clear()
        pass

    def print(self):
        """
        Print current state of global parameters for debugging.
        """
        print("stop_generate:", self.stop_generate)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)

