from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('words')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('words')

@dataclass
class Identity:
    """Enhanced identity information storage with linguistic features."""
    name: str
    role: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "explicit"
    references: Set[str] = field(default_factory=set)
    last_verified: float = field(default_factory=time.time)
    verification_count: int = 0
    linguistic_features: Dict = field(default_factory=dict)
    sentiment_score: float = 0.0
    context_words: List[str] = field(default_factory=list)

class IdentityManager:
    def __init__(self):
        """Initialize enhanced identity manager with NLP capabilities."""
        self.current_user = None
        self.assistant_name = "Brenda"
        self.assistant_identity = Identity(
            name=self.assistant_name,
            role="assistant",
            confidence=1.0,
            source="system",
            metadata={"type": "assistant", "permanent": True}
        )
        
        # Initialize NLP components
        self.sia = SentimentIntensityAnalyzer()
        
        # Track identity history with linguistic context
        self.identity_history = []
        self.verification_threshold = 0.7
        
        # Enhanced linguistic patterns
        self.name_patterns = self._compile_name_patterns()
        
        # Initialize excluded words with WordNet synonyms
        self.excluded_words = self._build_excluded_words()
        
    def _compile_name_patterns(self) -> List[Tuple[str, str, float]]:
        """Compile sophisticated name extraction patterns with confidence scores."""
        return [
            (r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "direct_statement", 1.0),
            (r"(?:this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "indirect_statement", 0.9),
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:here|speaking)", "contextual_introduction", 0.8),
            (r"(?:i go by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "alias_statement", 0.85),
            (r"(?:please call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "preference_statement", 0.95)
        ]

    def _build_excluded_words(self) -> Set[str]:
        """Build comprehensive set of excluded words using WordNet."""
        base_words = {
            "remember", "forget", "know", "tell", "say", "said", "called",
            "is", "am", "was", "were", "be", "being", "been",
            "the", "a", "an", "this", "that", "these", "those",
            "please", "thanks", "thank", "you", "me", "my", "your",
            "well", "good", "fine", "ok", "okay", "great", "alright",
            "hello", "hi", "hey", "assistant", "user", "system", "brenda"
        }
        
        # Add WordNet synonyms
        expanded_words = set()
        for word in base_words:
            synsets = wordnet.synsets(word)
            for synset in synsets:
                expanded_words.update(lemma.name().lower() for lemma in synset.lemmas())
        
        return base_words.union(expanded_words)

    def extract_name_from_text(self, text: str) -> Optional[Dict]:
        """Extract name using sophisticated NLP techniques."""
        try:
            if not text or not isinstance(text, str):
                return None
                
            text = text.strip()
            sentences = sent_tokenize(text)
            
            best_candidate = None
            max_confidence = 0.0
            
            for sentence in sentences:
                # Try pattern matching first
                for pattern, pattern_type, base_confidence in self.name_patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        name = match.group(1)
                        if self._validate_name(name):
                            confidence = base_confidence
                            # Adjust confidence based on sentence analysis
                            confidence *= self._analyze_sentence_confidence(sentence)
                            if confidence > max_confidence:
                                max_confidence = confidence
                                best_candidate = {
                                    "name": name,
                                    "confidence": confidence,
                                    "source": pattern_type,
                                    "context": sentence
                                }
            
            # Try NER if no good pattern match
            if not best_candidate or best_candidate["confidence"] < 0.8:
                ner_result = self._extract_name_with_ner(text)
                if ner_result and (not best_candidate or ner_result["confidence"] > best_candidate["confidence"]):
                    best_candidate = ner_result
            
            return best_candidate
            
        except Exception as e:
            print(f"Error extracting name: {e}")
            return None

    def _extract_name_with_ner(self, text: str) -> Optional[Dict]:
        """Extract name using NLTK's Named Entity Recognition."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            person_names = []
            
            if isinstance(named_entities, Tree):
                for subtree in named_entities.subtrees(filter=lambda t: t.label() == 'PERSON'):
                    name = ' '.join([leaf[0] for leaf in subtree.leaves()])
                    if self._validate_name(name):
                        person_names.append(name)
            
            if person_names:
                # Calculate confidence based on context
                confidence = 0.85  # Base confidence for NER
                context_score = self._analyze_name_context(text, person_names[0])
                final_confidence = min(1.0, confidence * context_score)
                
                return {
                    "name": person_names[0],
                    "confidence": final_confidence,
                    "source": "ner",
                    "context": text
                }
                
            return None
            
        except Exception as e:
            print(f"Error in NER extraction: {e}")
            return None

    def _analyze_sentence_confidence(self, sentence: str) -> float:
        """Analyze sentence structure for confidence scoring."""
        try:
            # Tokenize and tag
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            confidence = 1.0
            
            # Check sentence structure
            has_pronoun = any(tag.startswith('PRP') for _, tag in pos_tags)
            has_verb = any(tag.startswith('VB') for _, tag in pos_tags)
            has_proper_noun = any(tag == 'NNP' for _, tag in pos_tags)
            
            # Adjust confidence based on structure
            if not has_pronoun:
                confidence *= 0.9
            if not has_verb:
                confidence *= 0.8
            if not has_proper_noun:
                confidence *= 0.7
                
            # Consider sentiment
            sentiment = self.sia.polarity_scores(sentence)
            if sentiment['compound'] > 0:  # Positive sentiment
                confidence *= 1.1
                
            return min(1.0, confidence)
            
        except Exception as e:
            print(f"Error analyzing sentence: {e}")
            return 0.8  # Default confidence

    def _analyze_name_context(self, text: str, name: str) -> float:
        """Analyze contextual clues around the name."""
        try:
            # Tokenize and get window around name
            tokens = word_tokenize(text.lower())
            name_tokens = word_tokenize(name.lower())
            
            # Find name position
            name_indices = []
            for i in range(len(tokens) - len(name_tokens) + 1):
                if tokens[i:i+len(name_tokens)] == name_tokens:
                    name_indices.append(i)
            
            if not name_indices:
                return 1.0
            
            # Analyze context window
            context_score = 1.0
            window_size = 3
            
            for idx in name_indices:
                # Get context window
                start = max(0, idx - window_size)
                end = min(len(tokens), idx + len(name_tokens) + window_size)
                context = tokens[start:end]
                
                # Check for identity indicators
                if any(word in context for word in ["name", "call", "am", "is"]):
                    context_score *= 1.2
                    
                # Check for formal introductions
                if any(word in context for word in ["hello", "hi", "greetings"]):
                    context_score *= 1.1
                    
            return min(1.0, context_score)
            
        except Exception as e:
            print(f"Error analyzing context: {e}")
            return 1.0

    def _validate_name(self, name: str) -> bool:
        """Validate name using linguistic rules."""
        try:
            if not name:
                return False
                
            name_lower = name.lower()
            
            # Basic validation
            if (len(name) < 2 or  # Too short
                name_lower in self.excluded_words or  # Common words
                any(char.isdigit() for char in name) or  # Contains numbers
                not all(char.isalpha() or char.isspace() for char in name)):  # Non-letter chars
                return False
                
            # Check capitalization pattern
            words = name.split()
            if not all(word[0].isupper() and word[1:].islower() for word in words):
                return False
                
            # Check for reasonable length
            if any(len(word) > 20 for word in words):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating name: {e}")
            return False

    def parse_identity_statement(self, content: str, role: str) -> Optional[Dict]:
        """Parse identity from user statement using NLP."""
        try:
            if role != "user":
                return None
                
            # Extract name using NLP
            name_info = self.extract_name_from_text(content)
            if name_info:
                # Update current user with extracted name
                self.current_user = Identity(
                    name=name_info["name"],
                    role="user",
                    confidence=name_info["confidence"],
                    source=name_info["source"],
                    metadata={
                        "context": name_info["context"],
                        "extraction_method": name_info["source"]
                    }
                )
                
                # Add to identity history
                self.identity_history.append(self.current_user)
                
                return {
                    "type": "user",
                    "name": name_info["name"],
                    "confidence": name_info["confidence"],
                    "source": name_info["source"]
                }
                
            return None
            
        except Exception as e:
            print(f"Error parsing identity statement: {e}")
            return None