import unittest
from src.scraper import ImageScraper

class TestImageScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = ImageScraper()

    def test_fetch_image_urls(self):
        url = "http://example.com"
        images = self.scraper.fetch_image_urls(url)
        self.assertIsInstance(images, list)
        # Add more assertions based on expected behavior

    def test_save_images(self):
        images = ["image1.jpg", "image2.png"]
        save_directory = "./images"
        result = self.scraper.save_images(images, save_directory)
        self.assertTrue(result)
        # Add more assertions based on expected behavior

if __name__ == '__main__':
    unittest.main()