import glob
import os
import sys
from scraper import ImageScraper

def sanitize(name):
    """Simple sanitization for directory names."""
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip().replace(" ", "_")

def get_image_extension(url):
    """Returns a valid image extension extracted from the URL, defaulting to .jpg."""
    ext = os.path.splitext(url)[1]
    valid_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    if ext.lower() in valid_ext:
        return ext.lower()
    return '.jpg'

def load_existing_urls(search_dir):
    """Scans the search_dir for already downloaded images by reading url.txt files."""
    existing_urls = set()
    # Iterate over all subdirectories in the search directory.
    for folder in os.listdir(search_dir):
        folder_path = os.path.join(search_dir, folder)
        if os.path.isdir(folder_path):
            url_file = os.path.join(folder_path, "url.txt")
            if os.path.exists(url_file):
                with open(url_file, "r") as f:
                    # Assuming one URL per file.
                    url = f.read().strip()
                    if url:
                        existing_urls.add(url)
    return existing_urls

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_filename>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_dir = os.path.join(os.getcwd(), "img")
    os.makedirs(base_dir, exist_ok=True)

    scraper = ImageScraper()

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                term, count_str = line.split(",")
                search_term = term.strip()
                count = int(count_str.strip())
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

            search_dir = os.path.join(base_dir, sanitize(search_term))
            os.makedirs(search_dir, exist_ok=True)

            # Load already downloaded URLs for this search term.
            existing_urls = load_existing_urls(search_dir)

            # Fetch the image URLs for the search term.
            # Assuming fetch_images accepts a search term and returns URLs.
            image_urls = scraper.fetch_image_urls(search_term)
            
            success = 0
            for idx, image_url in enumerate(image_urls):
                # Skip this image if already downloaded.
                if image_url in existing_urls:
                    print(f"Skipping already downloaded image: {image_url}")
                    continue

                image_folder = os.path.join(search_dir, f"image_{idx+1}")
                os.makedirs(image_folder, exist_ok=True)

                try:
                    # Save the image. Assumes that the scraper saves the file into image_folder.
                    scraper.save_images([image_url], image_folder)
                    
                    # After saving, ensure the downloaded file gets the right extension.
                    files = glob.glob(os.path.join(image_folder, "*"))
                    if files:
                        # Assuming one file is downloaded per folder.
                        downloaded_file = files[0]
                        desired_ext = get_image_extension(image_url)
                        _, current_ext = os.path.splitext(downloaded_file)
                        if current_ext.lower() != desired_ext:
                            new_file = os.path.join(image_folder, f"{sanitize(search_term)}_{idx+1}{desired_ext}")
                            os.rename(downloaded_file, new_file)

                    # Write out the URL file for future checking.
                    url_file_path = os.path.join(image_folder, "url.txt")
                    with open(url_file_path, "w") as url_file:
                        url_file.write(image_url + "\n")
                    print(f"Saved image from {image_url} to {image_folder}")
                    success += 1
                except Exception as e:
                    print(f"Failed to save image from {image_url}: {e}")
                    continue
                if success >= count:
                    break

if __name__ == "__main__":
    main()