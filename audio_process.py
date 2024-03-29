import numpy as np
import librosa
from embedding import extract_vggish_embedding

def process_sound_file(audio_filename, sr=44100, top_db=10):
    audio_data, _ = librosa.load(audio_filename, sr=sr)
    spec_samples = int(16000*(10/1000) * 95 + 16000 * 25 / 1000)
    candidates = []
    for start_idx, end_idx in librosa.effects.split(audio_data, top_db=top_db):
        chunk = audio_data[start_idx:end_idx]
        onsets = librosa.onset.onset_detect(chunk)
        for onset_idx, onset_frame_idx in enumerate(onsets):
            onset_sample_idx = librosa.frames_to_samples(onset_frame_idx)[0]
            if onset_sample_idx < len(onsets) - 1:
                end_sample_idx = librosa.frames_to_samples(onsets[onset_sample_idx+1], sr=sr)[0]
            else:
                end_sample_idx = float('inf')

            end_sample_idx = min(min(end_sample_idx, onset_sample_idx + spec_samples), len(audio_data))

            if (end_sample_idx - onset_sample_idx) < spec_samples:
                pad_length = spec_samples - (end_sample_idx - onset_sample_idx)
                chunk_data = np.pad(audio_data[onset_sample_idx:end_sample_idx], (0, pad_length), 'constant')
            else:
                chunk_data = audio_data[onset_sample_idx:end_sample_idx]

            emb = extract_vggish_embedding(chunk_data, sr)
            assert emb.shape == (1, 128)
            emb = emb.reshape((128,))

            output_audio_start_idx = start_idx + onset_sample_idx
            output_audio_end_idx = start_idx + end_sample_idx

            cand_item = {
                'embedding': emb,
                'audio': output_audio_data[output_audio_start_idx, output_audio_end_idx],
                'start_idx': output_audio_start_idx,
                'end_idx': output_audio_end_idx
            }

            candidates.append(cand_item)
    return candidates
