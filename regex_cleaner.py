#!/usr/bin/env python3

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegexCleaner:
    def __init__(self, exclusion_files=None):
        self.patterns = [
            (r'\s+', ' '),  # replace multiple spaces with a single space
            (r'\s*-\s*', '-'),  # remove spaces around hyphens
            (r'\s*,\s*', ', '),  # remove spaces around commas
            (r'\s*\.\s*', '. '),  # remove spaces around periods
            (r'\s*\?\s*', '? '),  # remove spaces around question marks
            (r'\s*!\s*', '! '),  # remove spaces around exclamation marks
            (r'\s*;\s*', '; '),  # remove spaces around semicolons
            (r'\s*:\s*', ': '),  # remove spaces around colons
            (r'\s*\(\s*', ' ('),  # remove spaces around opening parentheses
            (r'\s*\)\s*', ') '),  # remove spaces around closing parentheses
            (r'\s*"\s*', '"'),  # remove spaces around quotation marks
            (r'\s*\'\s*', "'"),  # remove spaces around apostrophes
        ]
        self.exclusion_files = exclusion_files if exclusion_files else []

    def clean_text(self, text):
        original_length = len(text)
        if not text:
            logger.warning("Empty text provided to clean_text.")
            return text

        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)

        text = text.strip()
        cleaned_length = len(text)
        logger.info(f"Cleaned text from {original_length} to {cleaned_length} characters")
        return text
