# filepath: /home/sgthun/frizz/bus/README.md
# Image Scraper Project

This project is designed to scrape websites for images. It provides a simple interface to fetch images from a given URL and save them to a specified directory.

## Project Structure

```
bus
├── src
│   └── scraper.py        # Contains the main functionality for scraping images
├── tests
│   └── test_scraper.py   # Unit tests for the ImageScraper class
├── requirements.txt      # Lists the project dependencies
├── Dockerfile            # Instructions to build the Docker image
├── Makefile              # Commands to build and run the Docker container
└── README.md             # Project documentation
```

## Setup

1. Clone the repository.
2. Navigate to the `bus` directory.
3. Build the Docker image using the Makefile.
4. Run the scraper using the provided commands.

## Usage

To scrape images from a website, you can use the `ImageScraper` class defined in `src/scraper.py`. 

Example usage:

```python
from src.scraper import ImageScraper

scraper = ImageScraper()
images = scraper.fetch_images('http://example.com')
scraper.save_images(images, 'path/to/save/images')
```

## Running Tests

To run the unit tests, use the command defined in the Makefile.

## License

This project is licensed under the MIT License.