#!/usr/bin/env python3
"""
Advanced Video Dubbing Pipeline for TranscriptFusion
Uses enriched transcripts to generate precisely timed Spanish dubbing with Kokoro TTS

Usage:
    python dubbing_pipeline.py <video_id> --target-lang es --voice maria
    python dubbing_pipeline.py QkpqBCaUvS4 --target-lang es --voice isabella --speed-adjust 1.1

Features:
- Intelligent sentence grouping for natural speech patterns
- Speed adjustment to fit original timing
- Voice selection from Kokoro TTS models
- Automatic translation with context preservation
- Audio stretching/compression to match original timing
- Output synchronization with original video
"""

import sys
import os
import json
import re
import subprocess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
import tempfile
from pathlib import Path

try:
    import torch
    import torchaudio
    from scipy.io import wavfile
    import librosa
    import numpy as np
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("⚠️ Audio processing libraries not available. Install with: pip install torch torchaudio librosa scipy")

@dataclass
class DubSegment:
    """A segment of text to be dubbed with timing constraints"""
    start: float
    end: float
    original_text: str
    translated_text: str
    target_duration: float
    words: List[Dict] = None
    audio_file: str = None
    adjusted_speed: float = 1.0


@dataclass
class DubbingConfig:
    """Configuration for the dubbing process"""
    target_language: str = "es"  # Spanish
    voice_model: str = "ef_dora"   # Default Spanish female voice
    speed_adjustment: float = 1.0
    max_segment_duration: float = 8.0  # Max seconds per segment
    min_segment_duration: float = 1.0  # Min seconds per segment
    translation_service: str = "openai"  # "openai", "google", or "local"
    kokoro_endpoint: str = "http://localhost:8880"  # Your Kokoro TTS server
    pause_between_segments: float = 0.5  # Seconds


class TranslationService:
    """Handle translation of text segments - fully offline version"""
    
    def __init__(self, service_type: str = "local"):
        self.service_type = service_type
        self._load_local_translator()
    
    def _load_local_translator(self):
        """Load offline translation model"""
        try:
            # Try to load transformers-based translation
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-en-es"  # English to Spanish
            print(f"🔄 Loading offline translation model: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.translation_available = True
            print("✅ Offline translation model loaded successfully")
        except ImportError:
            print("⚠️ transformers library not available. Install with: pip install transformers")
            self.translation_available = False
        except Exception as e:
            print(f"⚠️ Could not load translation model: {e}")
            self.translation_available = False
    
    def translate_batch(self, texts: List[str], target_lang: str, context: str = "") -> List[str]:
        """Translate a batch of text segments maintaining context"""
        if self.service_type == "openai":
            return self._translate_with_openai(texts, target_lang, context)
        elif self.service_type == "google":
            return self._translate_with_google(texts, target_lang)
        else:
            return self._translate_local(texts, target_lang)
    
    def _translate_with_openai(self, texts: List[str], target_lang: str, context: str) -> List[str]:
        """Translate using OpenAI API with context awareness"""
        try:
            import openai
            
            # Check if API key is available
            if not os.getenv('OPENAI_API_KEY'):
                print("⚠️ No OpenAI API key found, falling back to local translation")
                return self._translate_local(texts, target_lang)
            
            # Create context-aware prompt
            text_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            
            prompt = f"""
            Translate the following English text segments to {target_lang}. 
            Maintain the natural flow and context between segments.
            Keep the emotional tone and speaking style appropriate for video dubbing.
            
            Context: {context}
            
            Text segments to translate:
            {text_list}
            
            Provide the translations in the same numbered format, preserving the meaning and natural speech patterns suitable for voice acting.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            translated_text = response.choices[0].message.content
            
            # Extract numbered translations
            translations = []
            for line in translated_text.split('\n'):
                if re.match(r'^\d+\.', line.strip()):
                    translation = re.sub(r'^\d+\.\s*', '', line.strip())
                    translations.append(translation)
            
            return translations if len(translations) == len(texts) else texts
            
        except Exception as e:
            print(f"OpenAI translation failed: {e}, falling back to local")
            return self._translate_local(texts, target_lang)
    
    def _translate_with_google(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate using Google Translate (free, no API key needed)"""
        try:
            from googletrans import Translator
            translator = Translator()
            
            translations = []
            print(f"🌐 Translating {len(texts)} segments with Google Translate...")
            
            for i, text in enumerate(texts):
                if i % 5 == 0:
                    print(f"   Progress: {i+1}/{len(texts)}")
                
                try:
                    result = translator.translate(text, dest=target_lang)
                    translations.append(result.text)
                except Exception as e:
                    print(f"   Failed to translate segment {i+1}: {e}")
                    translations.append(text)  # Fallback to original
            
            return translations
        except ImportError:
            print("⚠️ googletrans not available. Install with: pip install googletrans==4.0.0rc1")
            return self._translate_local(texts, target_lang)
        except Exception as e:
            print(f"Google translation failed: {e}, falling back to local")
            return self._translate_local(texts, target_lang)
    
    def _translate_local(self, texts: List[str], target_lang: str) -> List[str]:
        """Local translation using offline model"""
        if not self.translation_available:
            print("⚠️ No translation available - returning original text")
            print("💡 To enable translation:")
            print("   Option 1: pip install transformers torch (offline)")
            print("   Option 2: pip install googletrans==4.0.0rc1 (free online)")
            print("   Option 3: Set OPENAI_API_KEY environment variable (paid)")
            return texts
        
        try:
            print(f"🔄 Translating {len(texts)} segments with offline model...")
            translations = []
            
            for i, text in enumerate(texts):
                if i % 5 == 0:
                    print(f"   Progress: {i+1}/{len(texts)}")
                
                # Tokenize and translate
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                translations.append(translated)
            
            print("✅ Local translation completed")
            return translations
            
        except Exception as e:
            print(f"Local translation error: {e}")
            return texts


class KokoroTTSClient:
    """Interface for TTS synthesis - supports multiple offline engines"""
    
    def __init__(self, endpoint: str = "http://localhost:8880", fallback_engine: str = "pyttsx3"):
        self.endpoint = endpoint
        self.fallback_engine = fallback_engine
        self.use_kokoro = False
        self.available_voices = {
            "es": ["ef_dora", "em_alex", "em_santa"]  # Your actual Spanish voices
        }
        
        # Test Kokoro connection
        if self.test_kokoro_connection():
            self.use_kokoro = True
            print("✅ Kokoro TTS server connected")
        else:
            print("⚠️ Kokoro TTS not available, checking fallback options...")
            self._setup_fallback_tts()
    
    def _setup_fallback_tts(self):
        """Setup offline TTS fallback options"""
        # Try pyttsx3 (offline, cross-platform)
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            # Try to set Spanish voice if available
            voices = self.pyttsx3_engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    self.pyttsx3_engine.setProperty('voice', voice.id)
                    break
            self.fallback_available = True
            print("✅ pyttsx3 offline TTS ready")
            return
        except ImportError:
            print("   pyttsx3 not available")
        
        # Try gTTS (requires internet but no API key)
        try:
            from gtts import gTTS
            # Test with a small text
            test_tts = gTTS(text="test", lang='es', slow=False)
            self.fallback_available = True
            self.fallback_engine = "gtts"
            print("✅ gTTS (Google Text-to-Speech) ready - requires internet")
            return
        except ImportError:
            print("   gTTS not available")
        except Exception:
            print("   gTTS test failed (no internet?)")
        
        # Try espeak (Linux/Windows)
        try:
            result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.fallback_available = True
                self.fallback_engine = "espeak"
                print("✅ espeak system TTS ready")
                return
        except FileNotFoundError:
            print("   espeak not found")
        
        # Try festival (Linux)
        try:
            result = subprocess.run(['festival', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.fallback_available = True
                self.fallback_engine = "festival"
                print("✅ festival system TTS ready")
                return
        except FileNotFoundError:
            print("   festival not found")
        
        self.fallback_available = False
        print("❌ No TTS engines available")
        print("💡 Install options:")
        print("   pip install pyttsx3 (offline)")
        print("   pip install gtts (online, no API key)")
        print("   sudo apt install espeak (Linux)")
        print("   Or set up Kokoro TTS server")
    
    def synthesize(self, text: str, voice: str = "maria", speed: float = 1.0) -> Optional[str]:
        """Synthesize speech and return path to audio file"""
        if self.use_kokoro:
            return self._synthesize_kokoro(text, voice, speed)
        elif self.fallback_available:
            return self._synthesize_fallback(text, voice, speed)
        else:
            print("❌ No TTS engines available")
            return None
    
    def _synthesize_kokoro(self, text: str, voice: str, speed: float) -> Optional[str]:
        """Synthesize using your Kokoro API format"""
        try:
            payload = {
                "model": "kokoro",
                # "voice": voice,
                "voice": "ef_dora",
                # "input": "Hola, esto es una prueba",
                "input": text,
                "response_format": "mp3",
                # "speed": speed
                "speed": 1.0
            }
            
            print(f"🎤 Kokoro synthesis: '{text[:50]}{'...' if len(text) > 50 else ''}' (voice: {voice})")
            
            response = requests.post(
                f"{self.endpoint}/v1/audio/speech",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Kokoro API error {response.status_code}: {response.text}")
                return None
            
            if len(response.content) == 0:
                print("Kokoro API returned empty response")
                return None
            
            # Save MP3 response to temporary file
            temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_mp3.write(response.content)
            temp_mp3.close()
            
            print(f"✅ Kokoro audio generated: {len(response.content)} bytes")
            
            return temp_mp3.name
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to Kokoro API at localhost:8880")
            print("   Make sure your Kokoro service is running")
            return None
        except Exception as e:
            print(f"Kokoro API error: {e}")
            return None
    
    def _synthesize_fallback(self, text: str, voice: str, speed: float) -> Optional[str]:
        """Synthesize using fallback TTS engines"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            if self.fallback_engine == "pyttsx3":
                # Save to temporary file using pyttsx3
                self.pyttsx3_engine.setProperty('rate', int(200 * speed))
                self.pyttsx3_engine.save_to_file(text, temp_path)
                self.pyttsx3_engine.runAndWait()
                return temp_path
                
            elif self.fallback_engine == "gtts":
                from gtts import gTTS
                tts = gTTS(text=text, lang='es', slow=(speed < 0.8))
                tts.save(temp_path)
                return temp_path
                
            elif self.fallback_engine == "espeak":
                # Use espeak command line
                speed_param = int(150 * speed)  # espeak speed in words per minute
                command = [
                    'espeak', '-v', 'es', '-s', str(speed_param), 
                    '-w', temp_path, text
                ]
                subprocess.run(command, check=True, capture_output=True)
                return temp_path
                
            elif self.fallback_engine == "festival":
                # Use festival command line
                command = ['festival', '--tts', '--language', 'spanish']
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                process.communicate(input=text.encode())
                # Note: festival output would need additional processing
                return temp_path
                
        except Exception as e:
            print(f"Fallback TTS error ({self.fallback_engine}): {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None
    
    def test_kokoro_connection(self) -> bool:
        """Test if Kokoro TTS server is available"""
        try:
            # Test with a simple synthesis request
            payload = {
                "model": "kokoro",
                "voice": "ef_dora",
                "input": "test",
                "response_format": "mp3",
                "speed": 1.0
            }
            response = requests.post(
                f"{self.endpoint}/v1/audio/speech",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def test_connection(self) -> bool:
        """Test if any TTS engine is available"""
        return self.use_kokoro or self.fallback_available


class AudioProcessor:
    """Handle audio processing and synchronization"""
    
    @staticmethod
    def adjust_audio_duration(audio_file: str, target_duration: float, output_file: str) -> bool:
        """Adjust audio duration to match target timing using time stretching"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                print("Audio processing libraries not available")
                return False
            
            # Load audio
            y, sr = librosa.load(audio_file)
            current_duration = len(y) / sr
            
            # Calculate stretch factor
            stretch_factor = current_duration / target_duration
            
            # Apply time stretching
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
            
            # Save adjusted audio
            torchaudio.save(output_file, torch.tensor(y_stretched).unsqueeze(0), sr)
            return True
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return False
    
    @staticmethod
    def combine_audio_segments(segments: List[DubSegment], output_file: str) -> bool:
        """Combine all dubbed segments into a single audio track"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                print("Audio processing libraries not available")
                return False
            
            # Calculate total duration
            total_duration = max(seg.end for seg in segments)
            sample_rate = 22050  # Standard sample rate
            
            # Create silent audio track
            total_samples = int(total_duration * sample_rate)
            combined_audio = np.zeros(total_samples, dtype=np.float32)
            
            for segment in segments:
                if segment.audio_file and os.path.exists(segment.audio_file):
                    # Load segment audio
                    y, sr = librosa.load(segment.audio_file, sr=sample_rate)
                    
                    # Calculate position in combined track
                    start_sample = int(segment.start * sample_rate)
                    end_sample = start_sample + len(y)
                    
                    # Ensure we don't exceed array bounds
                    if end_sample <= total_samples:
                        combined_audio[start_sample:end_sample] = y
            
            # Save combined audio
            wavfile.write(output_file, sample_rate, combined_audio)
            return True
            
        except Exception as e:
            print(f"Audio combination error: {e}")
            return False


class VideoDubbingPipeline:
    """Main dubbing pipeline orchestrator"""
    
    def __init__(self, project_root: str = None, config: DubbingConfig = None):
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = script_dir 
        else:
            self.project_root = project_root
        
        self.config = config or DubbingConfig()
        self.transcripts_dir = os.path.join(self.project_root, "data", "transcripts")
        self.audio_dir = os.path.join(self.project_root, "data", "audio_clips")
        self.output_dir = os.path.join(self.project_root, "data", "dubbed")
        
        # Initialize services with fallback preferences
        self.translator = TranslationService("local")  # Default to offline
        self.tts_client = KokoroTTSClient(self.config.kokoro_endpoint)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🎬 Video Dubbing Pipeline initialized")
        print(f"📁 Project root: {self.project_root}")
        print(f"🗣️ Target language: {self.config.target_language}")
        print(f"🎙️ Voice model: {self.config.voice_model}")
    
    def load_transcript(self, video_id: str) -> Optional[Dict]:
        """Load the best available transcript (enriched > whisperx > official)"""
        transcript_files = [
            f"{video_id}_enriched.json",  # Best: enriched with human + whisperx
            f"{video_id}.json",           # Good: WhisperX with word timings
            f"{video_id}_official.json"   # Fallback: YouTube official
        ]
        
        for filename in transcript_files:
            filepath = os.path.join(self.transcripts_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ Loaded transcript: {filename}")
                    return {"data": data, "type": filename.split('_')[-1].split('.')[0]}
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    continue
        
        print(f"❌ No transcript found for video {video_id}")
        return None
    
    def create_dub_segments(self, transcript_data: Dict) -> List[DubSegment]:
        """Convert transcript to optimized dubbing segments"""
        segments = []
        transcript_type = transcript_data["type"]
        data = transcript_data["data"]
        
        if transcript_type in ["enriched", "json"]:  # WhisperX format
            source_segments = data.get("segments", [])
        else:  # Official YouTube format
            source_segments = data if isinstance(data, list) else []
        
        current_group = []
        current_start = None
        current_text = []
        
        for segment in source_segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", segment_start + segment.get("duration", 3))
            segment_text = segment.get("text", "").strip()
            
            if not segment_text:
                continue
            
            # Start new group if needed
            if current_start is None:
                current_start = segment_start
            
            current_group.append(segment)
            current_text.append(segment_text)
            
            # Check if we should close this group
            duration_so_far = segment_end - current_start
            should_close = (
                duration_so_far >= self.config.max_segment_duration or
                segment_text.endswith(('.', '!', '?')) or
                len(current_text) >= 3
            )
            
            if should_close and duration_so_far >= self.config.min_segment_duration:
                # Create segment
                combined_text = " ".join(current_text)
                segments.append(DubSegment(
                    start=current_start,
                    end=segment_end,
                    original_text=combined_text,
                    translated_text="",  # Will be filled later
                    target_duration=duration_so_far,
                    words=self._extract_words_from_group(current_group)
                ))
                
                # Reset for next group
                current_group = []
                current_start = None
                current_text = []
        
        # Handle remaining segments
        if current_group and current_text:
            combined_text = " ".join(current_text)
            segments.append(DubSegment(
                start=current_start,
                end=current_group[-1].get("end", current_start + 3),
                original_text=combined_text,
                translated_text="",
                target_duration=current_group[-1].get("end", current_start + 3) - current_start,
                words=self._extract_words_from_group(current_group)
            ))
        
        print(f"📝 Created {len(segments)} dubbing segments")
        return segments
    
    def _extract_words_from_group(self, segment_group: List[Dict]) -> List[Dict]:
        """Extract word-level timing data from a group of segments"""
        words = []
        for segment in segment_group:
            segment_words = segment.get("words", [])
            words.extend(segment_words)
        return words
    
    def translate_segments(self, segments: List[DubSegment]) -> List[DubSegment]:
        """Translate all segments maintaining context"""
        print(f"🌐 Translating {len(segments)} segments to {self.config.target_language}...")
        
        # Extract texts for batch translation
        texts = [seg.original_text for seg in segments]
        
        # Create context from all text
        full_context = " ".join(texts[:3])  # Use first few segments as context
        
        # Translate in batches
        batch_size = 10
        all_translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            translations = self.translator.translate_batch(
                batch, 
                self.config.target_language, 
                full_context
            )
            all_translations.extend(translations)
        
        # Update segments with translations
        for i, segment in enumerate(segments):
            if i < len(all_translations):
                segment.translated_text = all_translations[i]
            else:
                segment.translated_text = segment.original_text  # Fallback
        
        print("✅ Translation completed")
        return segments
    
    def synthesize_audio(self, segments: List[DubSegment]) -> List[DubSegment]:
        """Generate audio for all segments using Kokoro TTS"""
        print(f"🎙️ Synthesizing audio for {len(segments)} segments...")
        
        if not self.tts_client.test_connection():
            print("❌ Kokoro TTS server not available")
            return segments
        
        temp_dir = os.path.join(self.output_dir, "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            if not segment.translated_text:
                continue
            
            print(f"   Synthesizing segment {i+1}/{len(segments)}")
            
            # Calculate speed adjustment for timing
            estimated_words = len(segment.translated_text.split())
            estimated_duration = estimated_words * 0.6  # ~0.6 seconds per word average
            speed_factor = estimated_duration / segment.target_duration
            # final_speed = max(0.8, min(1.4, self.config.speed_adjustment * speed_factor)) # NEW: Clamp speed to reasonable range
            final_speed = 1.0  # Instead of calculated speed

            
            # Synthesize audio
            audio_file = self.tts_client.synthesize(
                segment.translated_text,
                self.config.voice_model,
                final_speed
            )
            
            if audio_file:
                segment.audio_file = audio_file
                segment.adjusted_speed = final_speed
                
                # # Adjust duration to match exactly
                # adjusted_file = os.path.join(temp_dir, f"segment_{i:04d}_adjusted.wav")
                # if AudioProcessor.adjust_audio_duration(audio_file, segment.target_duration, adjusted_file):
                #     segment.audio_file = adjusted_file
                #     segment.adjusted_speed = final_speed
                # else:
                #     segment.audio_file = audio_file
                
                # # Clean up temporary file
                # if os.path.exists(audio_file) and audio_file != segment.audio_file:
                #     os.unlink(audio_file)
        
        print("✅ Audio synthesis completed")
        return segments
    
    def create_final_audio(self, segments: List[DubSegment], video_id: str) -> str:
        """Combine all segments into final dubbed audio track"""
        print("🎵 Creating final dubbed audio track...")
        
        output_file = os.path.join(self.output_dir, f"{video_id}_dubbed_{self.config.target_language}.wav")
        
        if AudioProcessor.combine_audio_segments(segments, output_file):
            print(f"✅ Final audio saved: {output_file}")
            return output_file
        else:
            print("❌ Failed to create final audio")
            return ""
    
    def create_video_with_dubbed_audio(self, video_id: str, dubbed_audio_file: str) -> str:
        """Combine original video with dubbed audio using ffmpeg"""
        print("🎬 Creating final dubbed video...")
        
        original_video = os.path.join(self.audio_dir, f"{video_id}.mp4")
        if not os.path.exists(original_video):
            original_video = os.path.join(self.audio_dir, f"{video_id}.mkv")
        if not os.path.exists(original_video):
            original_video = os.path.join(self.audio_dir, f"{video_id}.webm")
        
        if not os.path.exists(original_video):
            print(f"❌ Original video not found for {video_id}")
            return ""
        
        output_video = os.path.join(self.output_dir, f"{video_id}_dubbed_{self.config.target_language}.mp4")
        
        command = [
            "ffmpeg",
            "-i", original_video,
            "-i", dubbed_audio_file,
            "-c:v", "copy",  # Copy video stream
            "-c:a", "aac",   # Encode audio as AAC
            "-map", "0:v:0", # Use video from first input
            "-map", "1:a:0", # Use audio from second input
            "-shortest",     # Match shortest stream
            "-y",           # Overwrite output
            output_video
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            print(f"✅ Dubbed video created: {output_video}")
            return output_video
        except subprocess.CalledProcessError as e:
            print(f"❌ Video creation failed: {e}")
            return ""
    
    def generate_report(self, video_id: str, segments: List[DubSegment]) -> str:
        """Generate a detailed report of the dubbing process"""
        report_file = os.path.join(self.output_dir, f"{video_id}_dubbing_report.txt")
        
        total_duration = max(seg.end for seg in segments) if segments else 0
        successful_segments = len([seg for seg in segments if seg.audio_file])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"VIDEO DUBBING REPORT\n")
            f.write(f"==================\n\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Target Language: {self.config.target_language}\n")
            f.write(f"Voice Model: {self.config.voice_model}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"SUMMARY\n")
            f.write(f"-------\n")
            f.write(f"Total segments: {len(segments)}\n")
            f.write(f"Successfully dubbed: {successful_segments}\n")
            f.write(f"Total duration: {total_duration:.2f} seconds\n")
            f.write(f"Success rate: {successful_segments/len(segments)*100:.1f}%\n\n")
            
            f.write(f"SEGMENT DETAILS\n")
            f.write(f"---------------\n")
            for i, seg in enumerate(segments):
                f.write(f"Segment {i+1}: [{seg.start:.2f}s - {seg.end:.2f}s]\n")
                f.write(f"  Original: {seg.original_text}\n")
                f.write(f"  Translated: {seg.translated_text}\n")
                f.write(f"  Duration: {seg.target_duration:.2f}s\n")
                f.write(f"  Speed adjustment: {seg.adjusted_speed:.2f}x\n")
                f.write(f"  Audio file: {'✅' if seg.audio_file else '❌'}\n\n")
        
        print(f"📊 Report saved: {report_file}")
        return report_file
    
    def process_video(self, video_id: str) -> bool:
        """Main processing pipeline for dubbing a video"""
        print(f"🎬 Starting dubbing process for video: {video_id}")
        print("=" * 60)
        
        # Load transcript
        transcript_data = self.load_transcript(video_id)
        if not transcript_data:
            return False
        
        # Create segments
        segments = self.create_dub_segments(transcript_data)
        if not segments:
            print("❌ No segments created")
            return False
        
        # Translate segments
        segments = self.translate_segments(segments)
        
        # Synthesize audio
        segments = self.synthesize_audio(segments)
        
        # Create final audio
        dubbed_audio = self.create_final_audio(segments, video_id)
        if not dubbed_audio:
            return False
        
        # Create final video (optional)
        dubbed_video = self.create_video_with_dubbed_audio(video_id, dubbed_audio)
        
        # Generate report
        self.generate_report(video_id, segments)
        
        print(f"\n🎉 Dubbing completed successfully!")
        print(f"🎵 Audio: {dubbed_audio}")
        if dubbed_video:
            print(f"🎬 Video: {dubbed_video}")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create dubbed video using TranscriptFusion enriched transcripts"
    )
    parser.add_argument('video_id', help='Video ID to dub')
    parser.add_argument('--target-lang', default='es', 
                       help='Target language code (default: es for Spanish)')
    parser.add_argument('--voice', default='ef_dora',
                       help='Voice model to use (default: ef_dora). Options: ef_dora, em_alex, em_santa')
    parser.add_argument('--speed-adjust', type=float, default=1.0,
                       help='Speed adjustment factor (default: 1.0)')
    parser.add_argument('--translation-service', default='openai',
                       choices=['openai', 'google', 'local'],
                       help='Translation service to use')
    parser.add_argument('--kokoro-endpoint', default='http://localhost:8880',
                       help='Kokoro TTS server endpoint')
    
    args = parser.parse_args()
    
    if not args.video_id:
        print("Usage: python dubbing_pipeline.py <video_id> [options]")
        print("Example: python dubbing_pipeline.py QkpqBCaUvS4 --target-lang es --voice maria")
        sys.exit(1)
    
    # Create configuration
    config = DubbingConfig(
        target_language=args.target_lang,
        voice_model=args.voice,
        speed_adjustment=args.speed_adjust,
        translation_service=args.translation_service,
        kokoro_endpoint=args.kokoro_endpoint
    )
    
    # Initialize pipeline
    pipeline = VideoDubbingPipeline(config=config)
    
    # Process video
    success = pipeline.process_video(args.video_id)
    
    if success:
        print("\n✨ Video dubbing completed successfully!")
    else:
        print("\n💥 Video dubbing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()