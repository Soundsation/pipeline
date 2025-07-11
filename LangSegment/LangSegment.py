import re

def filter_non_english_text(text):
    """
    Remove all non-English text (Chinese, Japanese, Korean, Thai, etc.)
    
    Args:
        text (str): Input text that may contain mixed languages
    Returns:
        str: Text with only English parts retained
    """
    if not text or text.strip() == "":
        return ""
    
    # Define patterns for non-English scripts to remove
    patterns = [
        (r'[\u4e00-\u9fff]+', ''),  # Chinese characters
        (r'[\u3040-\u309F\u30A0-\u30FF]+', ''),  # Japanese Hiragana and Katakana
        (r'[\uac00-\ud7a3]+', ''),  # Korean Hangul
        (r'[\u0E00-\u0E7F]+', ''),  # Thai
        (r'[А-Яа-яЁё]+', ''),  # Russian Cyrillic
        (r'[\u0600-\u06FF]+', '')  # Arabic
    ]
    
    # Apply each pattern to remove non-English scripts
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class EnglishLangSegment:
    """
    Simplified version of LangSegment that only processes English text
    """
    
    @staticmethod
    def getTexts(text):
        """
        Process text to retain only English parts
        
        Args:
            text (str): Input text that may contain mixed languages
        Returns:
            list: List containing only English text segments
        """
        if not text or text.strip() == "":
            return []
        
        # Filter out non-English characters
        english_only = filter_non_english_text(text)
        
        # If there's no text left after filtering, return empty list
        if not english_only:
            return []
        
        # Return the English text as a single segment
        return [{"lang": "en", "text": english_only, "score": 1.0}]
    
    @staticmethod
    def getCounts():
        """
        Return language statistics (always English)
        
        Returns:
            list: Single-element list with English count
        """
        return [("en", 1)]

# Replace the original LangSegment with our English-only version
LangSegment = EnglishLangSegment

# Keep the original function interfaces for compatibility
def getTexts(text):
    return LangSegment.getTexts(text)

def getCounts():
    return LangSegment.getCounts()
