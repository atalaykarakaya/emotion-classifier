# test_labels.py olarak kaydet, dataset klasörünün yanına koy
from feature_extraction import get_label_from_filename, VALID_EMOTIONS
import glob, os

wav_files = glob.glob("dataset/**/*.wav", recursive=True)
bad = []
labels = {}

for f in wav_files:
    label = get_label_from_filename(f)
    if label is None:
        bad.append(os.path.basename(f))
    else:
        labels[label] = labels.get(label, 0) + 1

print("Gecerli siniflar ve sayilari:")
for k, v in sorted(labels.items()):
    print(f"  {k}: {v}")
print(f"\nSkip edilecek dosya sayisi: {len(bad)}")
if bad:
    print("Örnekler:", bad[:5])