# VoiceWeave

**Video dubbing pipeline that creates multilingual audio using enriched transcripts and Kokoro TTS with timing adjustments.**

VoiceWeave takes word-level timing data from [TranscriptFusion](https://github.com/daniel-carvajal/TranscriptFusion) and generates Spanish dubbing with timing corrections to improve synchronization.

## Features

- **Transcript Processing** - Uses enriched transcripts from TranscriptFusion with word-level timing
- **Sentence Grouping** - Combines words into natural speech segments
- **Offline Translation** - Helsinki-NLP models for English → Spanish translation
- **TTS Integration** - [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) with speed adjustment capabilities
- **Timing Corrections** - Retry mechanism and overlap detection to improve synchronization
- **Voice Options** - Spanish voices: ef_dora, em_alex, em_santa

## Quick Start

### Prerequisites

- Python 3.8+
- [TranscriptFusion](https://github.com/daniel-carvajal/TranscriptFusion) for enriched transcripts
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) server running on localhost:8880

### Installation

```bash
git clone https://github.com/daniel-carvajal/VoiceWeave.git
cd VoiceWeave
pip install -r requirements.txt
```

### Setup Kokoro TTS

1. Follow the setup instructions at [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
2. Start the Kokoro server on localhost:8880:
   ```bash
   # Using Docker (recommended)
   docker run -p 8880:8880 kokoro-fastapi
   
   # Or run directly
   ./start-cpu.sh  # or start-gpu.sh for GPU acceleration
   ```

### Basic Usage

```bash
# Create enriched transcript first (TranscriptFusion)
cd ../TranscriptFusion
