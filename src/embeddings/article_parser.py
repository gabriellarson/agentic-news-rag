"""
Article parser for structured news text files.

Parses articles with the following format:
Title: [Article Title]
Subtitle: [Article Subtitle]
Authors: [Comma-separated authors or empty]
Published: [ISO 8601 timestamp]

[Article body text...]
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Parsed article data structure"""
    title: str
    subtitle: Optional[str]
    authors: List[str]
    published: datetime
    content: str
    file_path: Path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'authors': self.authors,
            'published': self.published.isoformat(),
            'content': self.content,
            'file_path': str(self.file_path)
        }
    
    @property
    def full_text(self) -> str:
        """Get full searchable text (title + subtitle + content)"""
        parts = [self.title]
        if self.subtitle:
            parts.append(self.subtitle)
        parts.append(self.content)
        return '\n\n'.join(parts)
    
    @property
    def metadata(self) -> Dict:
        """Get metadata for vector storage"""
        return {
            'title': self.title,
            'subtitle': self.subtitle or '',
            'authors': ', '.join(self.authors) if self.authors else '',
            'published': self.published.isoformat(),
            'file_path': str(self.file_path)
        }


class ArticleParser:
    """Parser for structured news article text files"""
    
    def __init__(self):
        self.header_pattern = re.compile(
            r'^Title:\s*(.+?)\n'
            r'Subtitle:\s*(.*?)\n'
            r'Authors:\s*(.*?)\n'
            r'Published:\s*(.+?)\n\n',
            re.MULTILINE | re.DOTALL
        )
    
    def parse_file(self, file_path: Path) -> Article:
        """Parse a single article file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.parse_text(content, file_path)
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise
    
    def parse_text(self, text: str, file_path: Path = None) -> Article:
        """Parse article text content"""
        match = self.header_pattern.match(text)
        
        if not match:
            raise ValueError(f"Invalid article format in {file_path or 'text'}")
        
        title = match.group(1).strip()
        subtitle = match.group(2).strip() or None
        authors_str = match.group(3).strip()
        published_str = match.group(4).strip()
        
        # Parse authors
        authors = []
        if authors_str:
            authors = [author.strip() for author in authors_str.split(',')]
        
        # Parse published date
        try:
            published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
        except ValueError:
            # Try alternative formats if needed
            logger.warning(f"Could not parse date '{published_str}', using current time")
            published = datetime.now()
        
        # Extract body content (everything after the header)
        body_start = match.end()
        content = text[body_start:].strip()
        
        return Article(
            title=title,
            subtitle=subtitle,
            authors=authors,
            published=published,
            content=content,
            file_path=file_path or Path('unknown')
        )
    
    def parse_directory(self, directory: Path, 
                       pattern: str = "*.txt") -> Iterator[Article]:
        """Parse all articles in a directory"""
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"Directory {directory} does not exist")
        
        article_files = sorted(directory.glob(pattern))
        logger.info(f"Found {len(article_files)} articles in {directory}")
        
        for file_path in article_files:
            try:
                article = self.parse_file(file_path)
                yield article
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                continue
    
    def validate_article(self, article: Article) -> List[str]:
        """Validate parsed article and return any issues"""
        issues = []
        
        if not article.title:
            issues.append("Missing title")
        
        if not article.content:
            issues.append("Missing content")
        
        # Compare with timezone-aware current time
        current_time = datetime.now(timezone.utc)
        if article.published.tzinfo:
            # Article has timezone info
            if article.published > current_time:
                issues.append("Published date is in the future")
        else:
            # Article is timezone-naive, compare without timezone
            if article.published > current_time.replace(tzinfo=None):
                issues.append("Published date is in the future")
        
        if len(article.content) < 100:
            issues.append("Content seems too short")
        
        return issues


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize parser
    parser = ArticleParser()
    
    # Test with a single file
    test_file = Path(__file__).parent.parent.parent / "text_articles" / "Chesapeake and Southwestern to create US gas titan with $7.4bn deal.txt"
    
    if test_file.exists():
        print(f"Parsing test file: {test_file}")
        article = parser.parse_file(test_file)
        
        print(f"\nTitle: {article.title}")
        print(f"Subtitle: {article.subtitle}")
        print(f"Authors: {', '.join(article.authors)}")
        print(f"Published: {article.published}")
        print(f"Content length: {len(article.content)} chars")
        print(f"First 200 chars: {article.content[:200]}...")
        
        # Validate
        issues = parser.validate_article(article)
        if issues:
            print(f"\nValidation issues: {issues}")
        else:
            print("\nArticle validated successfully!")
    
    # Test parsing all articles
    articles_dir = Path(__file__).parent.parent.parent / "text_articles"
    if articles_dir.exists():
        print(f"\n\nParsing all articles in {articles_dir}")
        articles = list(parser.parse_directory(articles_dir))
        print(f"Successfully parsed {len(articles)} articles")
        
        for article in articles[:3]:  # Show first 3
            print(f"\n- {article.title}")
            print(f"  Published: {article.published.date()}")
            print(f"  Authors: {', '.join(article.authors) or 'None'}")