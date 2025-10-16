/**
 * AudioProcessor - A custom AudioWorkletProcessor for real-time audio recording
 * Captures audio input and converts it to Int16 format for transmission
 */
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Flag to control whether audio recording is active
    this.isRecording = false;
    
    // Listen for messages from the main thread to control recording state
    this.port.onmessage = (event) => {
      if (event.data.command === 'setRecording') {
        this.isRecording = event.data.value;
      }
    };
  }

  /**
   * Process audio data in real-time
   * @param {Float32Array[][]} inputs - Input audio data from the microphone
   * @param {Float32Array[][]} outputs - Output audio data (not used in this processor)
   * @param {Object} parameters - Audio parameters (not used in this processor)
   * @returns {boolean} - Returns true to keep the processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const inputChannel = input[0];
    
    // Only process audio when recording is active and input data is available
    if (this.isRecording && inputChannel) {
      // Convert Float32 audio samples to Int16 format
      const int16Array = new Int16Array(inputChannel.length);
      for (let i = 0; i < inputChannel.length; i++) {
        // Scale from [-1.0, 1.0] to [-32768, 32767]
        int16Array[i] = inputChannel[i] * 0x7FFF;
      }
      
      // Send the converted audio data back to the main thread
      this.port.postMessage({
        audio: Array.from(new Uint8Array(int16Array.buffer)), // Int16 data as bytes
        inputData: Array.from(inputChannel) // Original Float32 data for debugging/visualization
      });
    }
    
    // Return true to keep the processor running
    return true;
  }
}

// Register this processor with the name 'audio-processor'
registerProcessor('audio-processor', AudioProcessor);
