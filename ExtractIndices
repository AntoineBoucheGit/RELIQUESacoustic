from maad import sound, features
import pandas as pd
import os
import soundfile as sf
import tempfile

def get_flac_files(root_dir):
    flac_files = []
    for dirpath, _, filenames in os.walk(root_dir):  
        for file in filenames:
            if file.lower().endswith(".flac"): 
                flac_files.append(os.path.join(dirpath, file))
    return flac_files




def process_audio_files(root_dir):
    flac_files = get_flac_files(root_dir)
    indices_tot=[]
    for file in flac_files: 
        temp_wav_file = None
        try:
            
            nom_fichier = os.path.splitext(os.path.basename(file))[0]  
            parts = nom_fichier.replace('_', '-').split('-')
            if len(parts) < 6:
                print(f"Format invalide pour {file}, ignoré.")
                continue

            nom_zone = parts[0]  
            distance_lisiere = parts[1]   
            date = parts[4]  
            heure = parts[5] 
            
            # FLAC -> WAV 
            data, samplerate = sf.read(file)
            temp_dir = tempfile.gettempdir()
            temp_wav_file = os.path.join(temp_dir, f"{nom_fichier}_temp.wav")
            sf.write(temp_wav_file, data, samplerate)
           
            # Load audio
            s, fs = sound.load(temp_wav_file) 
            duration = len(s) / fs  

            if duration < 60:
                print(f"File too short ({duration:.2f} sec) : {file}, ignoré.")
                continue  

            s = s[:60 * fs]  # Keep only the first minute 

            # Calcul du spectrogramme
            Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, window='hann', nperseg=1024, noverlap=512, db=False)
            
            # Calcul des indices acoustiques
            spectral_results = features.all_spectral_alpha_indices(Sxx_power, tn, fn, display=False)
            temporal_results = features.all_temporal_alpha_indices(s, fs, display=False)
            
            spectral_indices = spectral_results[0] if isinstance(spectral_results, tuple) else spectral_results
            temporal_indices = temporal_results[0] if isinstance(temporal_results, tuple) else temporal_results
            
            # Fusion des indices 
            indices = {
                'Fichier': os.path.basename(file),
                'Zone': nom_zone,
                'Distance_lisiere': distance_lisiere,
                'Heure': heure,
                'Date': date
            }
            indices.update(spectral_indices.to_dict(orient='records')[0])  
            indices.update(temporal_indices.to_dict(orient='records')[0])
            
            indices_tot.append(indices)
            
          
        
        except Exception as e:
            print(f"Error with file {file}: {e}")
        finally:
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.remove(temp_wav_file)
                except Exception as e:
                    print(f"Error deleting temporary file {temp_wav_file}: {e}")
    
    return pd.DataFrame(indices_tot)



# Root
root_directory = "root"
csv_path = os.path.join(root_directory, "indices_tot.csv")

# Exec
df_indices = process_audio_files(root_directory)

csv_path = os.path.join(root_directory, "indices_tot.csv")

# Save df->csv
df_indices.to_csv(csv_path, index=False, sep=";")

print(f"Saved file : {csv_path}")
