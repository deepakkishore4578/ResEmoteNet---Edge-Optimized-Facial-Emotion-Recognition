import os
import zipfile

# Make sure you have kaggle installed: pip install kaggle
# And your kaggle.json is in ~/.kaggle/

def setup_data():
    print("Downloading FER-2013 (Image Version)...")
    # This specific dataset has the folder structure we need (msambare/fer2013)
    os.system("kaggle datasets download -d msambare/fer2013")
    
    print("Unzipping...")
    with zipfile.ZipFile("fer2013.zip", 'r') as zip_ref:
        zip_ref.extractall("temp_data")
    
    # Move files to correct structure
    # The zip usually extracts to 'train' and 'test' folders directly
    if os.path.exists("data"):
        import shutil
        shutil.rmtree("data")
        
    os.rename("temp_data", "data")
    
    # Clean up
    os.remove("fer2013.zip")
    print("Data Setup Complete!")
    print("Checking 'neutral' folder count:")
    print(len(os.listdir("data/train/neutral")))

if __name__ == "__main__":
    setup_data()