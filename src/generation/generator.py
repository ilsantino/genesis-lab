"""
Generators for synthetic data creation using AWS Bedrock.

This module provides:
- BaseGenerator: Abstract base class for all generators
- CustomerServiceGenerator: Generates Banking77-style customer service conversations
"""

import json
import logging
import random
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.aws_client import BedrockClient
from src.utils.config import get_config, DomainType
from src.utils.config.schemas import GenerationParams

from .templates.customer_service_prompts import (
    ALL_INTENTS,
    SYSTEM_PROMPTS,
    Sentiment,
    Complexity,
    Language,
    EmotionArc,
    DEFAULT_DISTRIBUTION,
    build_full_prompt_with_examples,
    validate_conversation_schema,
    INTENT_TO_CATEGORY,
)


__all__ = ["BaseGenerator", "CustomerServiceGenerator"]

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for synthetic data generators.
    
    Provides common functionality for all domain-specific generators:
    - BedrockClient integration
    - Configuration management
    - Logging and metrics
    
    Subclasses must implement:
    - generate_single(): Generate a single data item
    - generate_batch(): Generate multiple data items
    """
    
    def __init__(
        self,
        client: BedrockClient,
        domain: DomainType,
        generation_params: Optional[GenerationParams] = None
    ):
        """
        Initialize the generator.
        
        Args:
            client: BedrockClient instance for LLM calls
            domain: Domain type identifier (e.g., "customer_service")
            generation_params: Optional override for generation parameters
        """
        self.client = client
        self.domain = domain
        self._config = get_config()
        
        # Get domain-specific config
        self._domain_config = self._config.get_domain_config(domain)
        
        # Use provided params or fall back to domain config
        self._generation_params = generation_params or self._domain_config.generation_params
        
        # Metrics
        self._total_generated = 0
        self._total_failed = 0
        
        logger.info(
            f"{self.__class__.__name__} initialized for domain '{domain}' "
            f"(temperature={self._generation_params.temperature}, "
            f"max_tokens={self._generation_params.max_tokens})"
        )
    
    @classmethod
    def from_config(cls, domain: DomainType) -> "BaseGenerator":
        """
        Create generator from global configuration.
        
        Args:
            domain: Domain type identifier
        
        Returns:
            Configured generator instance
        """
        client = BedrockClient.from_config()
        return cls(client=client, domain=domain)
    
    @abstractmethod
    def generate_single(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a single data item.
        
        Args:
            **kwargs: Domain-specific generation parameters
        
        Returns:
            Generated data as dictionary
        
        Raises:
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        count: int,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple data items.
        
        Args:
            count: Number of items to generate
            **kwargs: Domain-specific generation parameters
        
        Returns:
            List of generated data dictionaries
        """
        pass
    
    def _build_system_prompt(self, language: str = "en") -> str:
        """
        Build system prompt for the domain.
        
        Args:
            language: Language code (e.g., "en", "es")
        
        Returns:
            System prompt string
        """
        # Default implementation - subclasses can override
        return f"You are an expert at generating synthetic {self.domain} data."
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with fallback handling.
        
        Args:
            response: Raw LLM response string
        
        Returns:
            Parsed JSON as dictionary
        
        Raises:
            ValueError: If JSON parsing fails after all attempts
        """
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try extracting JSON array from markdown
        array_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
        if array_match:
            try:
                return json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding raw JSON object
        obj_match = re.search(r'\{[\s\S]*\}', response)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try finding raw JSON array
        arr_match = re.search(r'\[[\s\S]*\]', response)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to parse JSON from response: {response[:200]}...")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get generator metrics."""
        return {
            "total_generated": self._total_generated,
            "total_failed": self._total_failed,
            "success_rate": (
                self._total_generated / (self._total_generated + self._total_failed)
                if (self._total_generated + self._total_failed) > 0
                else 0.0
            )
        }


class CustomerServiceGenerator(BaseGenerator):
    """
    Generator for Banking77-style customer service conversations.
    
    Generates realistic multi-turn dialogues between customers and support agents
    for neobank/fintech applications.
    
    Features:
    - 77 Banking77 intents support
    - Bilingual (English/Spanish)
    - Configurable sentiment, complexity, emotion arcs
    - Few-shot prompting for quality
    
    Example:
        >>> client = BedrockClient.from_config()
        >>> generator = CustomerServiceGenerator(client, "customer_service")
        >>> conversation = generator.generate_single(intent="card_arrival")
        >>> print(conversation["turns"][0]["text"])
    """
    
    def __init__(
        self,
        client: BedrockClient,
        domain: DomainType = "customer_service",
        generation_params: Optional[GenerationParams] = None
    ):
        """
        Initialize CustomerServiceGenerator.
        
        Args:
            client: BedrockClient for LLM calls
            domain: Domain type (should be "customer_service")
            generation_params: Optional generation parameter overrides
        """
        super().__init__(client, domain, generation_params)
        
        # Available intents from Banking77
        self._intents = ALL_INTENTS
        
        logger.info(f"CustomerServiceGenerator ready with {len(self._intents)} intents")
    
    @classmethod
    def from_config(cls) -> "CustomerServiceGenerator":
        """Create CustomerServiceGenerator from global configuration."""
        client = BedrockClient.from_config()
        return cls(client=client, domain="customer_service")
    
    def _build_system_prompt(self, language: str = "en") -> str:
        """
        Build system prompt for customer service conversations.
        
        Args:
            language: Language code ("en" or "es")
        
        Returns:
            System prompt string
        """
        return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
    
    def _select_random_parameters(
        self,
        intent: Optional[str] = None,
        sentiment: Optional[str] = None,
        complexity: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select random parameters based on distribution config.
        
        Args:
            intent: Optional specific intent (random if None)
            sentiment: Optional specific sentiment (random if None)
            complexity: Optional specific complexity (random if None)
            language: Optional specific language (random if None)
        
        Returns:
            Dictionary with selected parameters
        """
        dist = DEFAULT_DISTRIBUTION
        
        # Select intent
        selected_intent = intent or random.choice(self._intents)
        
        # Select sentiment based on distribution
        if sentiment:
            selected_sentiment = sentiment
        else:
            sentiments = list(dist["sentiment"].keys())
            weights = list(dist["sentiment"].values())
            selected_sentiment = random.choices(sentiments, weights=weights)[0]
        
        # Select complexity based on distribution
        if complexity:
            selected_complexity = complexity
        else:
            complexities = list(dist["complexity"].keys())
            weights = list(dist["complexity"].values())
            selected_complexity = random.choices(complexities, weights=weights)[0]
        
        # Select language based on distribution
        if language:
            selected_language = language
        else:
            languages = list(dist["language"].keys())
            weights = list(dist["language"].values())
            selected_language = random.choices(languages, weights=weights)[0]
        
        return {
            "intent": selected_intent,
            "sentiment": selected_sentiment,
            "complexity": selected_complexity,
            "language": selected_language
        }
    
    def generate_single(
        self,
        intent: Optional[str] = None,
        sentiment: Optional[str] = None,
        complexity: Optional[str] = None,
        language: str = "en",
        emotion_arc: Optional[str] = None,
        num_examples: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a single customer service conversation.
        
        Args:
            intent: Banking77 intent (random if None)
            sentiment: "positive", "neutral", or "negative" (random if None)
            complexity: "simple", "medium", or "complex" (random if None)
            language: "en" or "es" (default: "en")
            emotion_arc: Optional emotion arc type
            num_examples: Number of few-shot examples to include
        
        Returns:
            Generated conversation dictionary with structure:
            {
                "conversation_id": str,
                "intent": str,
                "category": str,
                "sentiment": str,
                "complexity": str,
                "language": str,
                "turn_count": int,
                "customer_emotion_arc": str,
                "resolution_time_category": str,
                "resolution_status": str,
                "turns": [{"speaker": str, "text": str}, ...],
                "metadata": {...}
            }
        
        Raises:
            RuntimeError: If generation fails after retries
            ValueError: If invalid parameters provided
        """
        # Select parameters (random where not specified)
        params = self._select_random_parameters(
            intent=intent,
            sentiment=sentiment,
            complexity=complexity,
            language=language
        )
        
        selected_intent = params["intent"]
        selected_sentiment = params["sentiment"]
        selected_complexity = params["complexity"]
        selected_language = params["language"]
        
        # Convert to enums for prompt builder
        sentiment_enum = Sentiment(selected_sentiment)
        complexity_enum = Complexity(selected_complexity)
        language_enum = Language(selected_language)
        emotion_arc_enum = EmotionArc(emotion_arc) if emotion_arc else None
        
        # Build prompt with few-shot examples
        prompts = build_full_prompt_with_examples(
            intent=selected_intent,
            sentiment=sentiment_enum,
            complexity=complexity_enum,
            language=language_enum,
            num_examples=num_examples,
            emotion_arc=emotion_arc_enum
        )
        
        try:
            # Call LLM
            response = self.client.invoke_model(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=self._generation_params.temperature,
                max_tokens=self._generation_params.max_tokens,
                top_p=self._generation_params.top_p
            )
            
            # Parse response
            conversation = self._parse_json_response(response)
            
            # Validate schema
            errors = validate_conversation_schema(conversation)
            if errors:
                logger.warning(f"Validation warnings: {errors}")
                # Try to fix common issues
                conversation = self._fix_conversation_schema(
                    conversation,
                    selected_intent,
                    selected_sentiment,
                    selected_complexity,
                    selected_language
                )
            
            # Add metadata
            conversation["metadata"] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": self._config.aws.default_model,
                "generator_version": "1.0.0"
            }
            
            # Ensure conversation_id is unique
            if not conversation.get("conversation_id") or conversation["conversation_id"].startswith("conv_XXX"):
                conversation["conversation_id"] = f"conv_{uuid.uuid4().hex[:12]}"
            
            self._total_generated += 1
            logger.debug(f"Generated conversation: {conversation['conversation_id']}")
            
            return conversation
            
        except Exception as e:
            self._total_failed += 1
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate conversation: {e}") from e
    
    def _fix_conversation_schema(
        self,
        conversation: Dict[str, Any],
        intent: str,
        sentiment: str,
        complexity: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Attempt to fix common schema issues in generated conversations.
        
        Args:
            conversation: The conversation dict to fix
            intent: Expected intent
            sentiment: Expected sentiment
            complexity: Expected complexity
            language: Expected language
        
        Returns:
            Fixed conversation dictionary
        """
        # Ensure required fields exist
        if "conversation_id" not in conversation:
            conversation["conversation_id"] = f"conv_{uuid.uuid4().hex[:12]}"
        
        if "intent" not in conversation:
            conversation["intent"] = intent
        
        if "category" not in conversation:
            conversation["category"] = INTENT_TO_CATEGORY.get(intent, "general")
        
        if "sentiment" not in conversation:
            conversation["sentiment"] = sentiment
        
        if "complexity" not in conversation:
            conversation["complexity"] = complexity
        
        if "language" not in conversation:
            conversation["language"] = language
        
        if "turns" in conversation:
            conversation["turn_count"] = len(conversation["turns"])
        elif "turn_count" not in conversation:
            conversation["turn_count"] = 0
        
        if "customer_emotion_arc" not in conversation:
            # Infer from sentiment
            if sentiment == "positive":
                conversation["customer_emotion_arc"] = "stable_positive"
            elif sentiment == "negative":
                conversation["customer_emotion_arc"] = "frustrated_to_satisfied"
            else:
                conversation["customer_emotion_arc"] = "stable_neutral"
        
        if "resolution_time_category" not in conversation:
            # Infer from complexity
            mapping = {"simple": "quick", "medium": "standard", "complex": "extended"}
            conversation["resolution_time_category"] = mapping.get(complexity, "standard")
        
        if "resolution_status" not in conversation:
            conversation["resolution_status"] = "resolved"
        
        return conversation
    
    def generate_batch(
        self,
        count: int,
        intents: Optional[List[str]] = None,
        sentiment_distribution: Optional[Dict[str, float]] = None,
        complexity_distribution: Optional[Dict[str, float]] = None,
        language: str = "en",
        continue_on_error: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple customer service conversations.
        
        Args:
            count: Number of conversations to generate
            intents: Optional list of intents to use (cycled if fewer than count)
            sentiment_distribution: Optional custom sentiment distribution
            complexity_distribution: Optional custom complexity distribution
            language: Target language ("en" or "es")
            continue_on_error: If True, continue generating even if some fail
        
        Returns:
            List of generated conversation dictionaries
        
        Raises:
            RuntimeError: If generation fails and continue_on_error is False
        """
        results: List[Dict[str, Any]] = []
        
        # Prepare intent list
        if intents:
            # Cycle through provided intents
            intent_cycle = [intents[i % len(intents)] for i in range(count)]
        else:
            # Random intents
            intent_cycle = [None] * count
        
        # Custom distributions
        sent_dist = sentiment_distribution or DEFAULT_DISTRIBUTION["sentiment"]
        comp_dist = complexity_distribution or DEFAULT_DISTRIBUTION["complexity"]
        
        for i in range(count):
            logger.info(f"Generating conversation {i + 1}/{count}...")
            
            # Select sentiment based on distribution
            sentiments = list(sent_dist.keys())
            weights = list(sent_dist.values())
            selected_sentiment = random.choices(sentiments, weights=weights)[0]
            
            # Select complexity based on distribution
            complexities = list(comp_dist.keys())
            weights = list(comp_dist.values())
            selected_complexity = random.choices(complexities, weights=weights)[0]
            
            try:
                conversation = self.generate_single(
                    intent=intent_cycle[i],
                    sentiment=selected_sentiment,
                    complexity=selected_complexity,
                    language=language
                )
                results.append(conversation)
                
            except Exception as e:
                logger.error(f"Failed to generate conversation {i + 1}: {e}")
                if not continue_on_error:
                    raise
        
        logger.info(
            f"Batch generation complete: {len(results)}/{count} successful "
            f"({self._total_failed} total failures)"
        )
        
        return results
    
    def generate_balanced_dataset(
        self,
        total_count: int,
        language: str = "en",
        include_all_intents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a balanced dataset across intents, sentiments, and complexities.
        
        This method ensures good coverage across all dimensions for training data.
        
        Args:
            total_count: Total number of conversations to generate
            language: Target language ("en" or "es")
            include_all_intents: If True, ensures all 77 intents are represented
        
        Returns:
            List of generated conversations with balanced distribution
        """
        results: List[Dict[str, Any]] = []
        
        if include_all_intents:
            # Calculate how many per intent (minimum 1 each)
            per_intent = max(1, total_count // len(self._intents))
            remaining = total_count - (per_intent * len(self._intents))
            
            # Generate for each intent
            for intent in self._intents:
                count_for_intent = per_intent
                if remaining > 0:
                    count_for_intent += 1
                    remaining -= 1
                
                intent_results = self.generate_batch(
                    count=count_for_intent,
                    intents=[intent],
                    language=language,
                    continue_on_error=True
                )
                results.extend(intent_results)
        else:
            # Just generate with default distribution
            results = self.generate_batch(
                count=total_count,
                language=language,
                continue_on_error=True
            )
        
        logger.info(f"Generated balanced dataset: {len(results)} conversations")
        return results
    
    @property
    def available_intents(self) -> List[str]:
        """Get list of available Banking77 intents."""
        return list(self._intents)
    
    @property
    def intent_count(self) -> int:
        """Get number of available intents."""
        return len(self._intents)
    
    # =========================================================================
    # BATCH INFERENCE METHODS
    # =========================================================================
    
    def prepare_batch_prompts(
        self,
        count: int,
        language: str = "en",
        intents: Optional[List[str]] = None,
        sentiment_distribution: Optional[Dict[str, float]] = None,
        complexity_distribution: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare prompts for batch inference without calling the API.
        
        This method builds all prompts that would be used for generation,
        suitable for submission to batch inference.
        
        Args:
            count: Number of prompts to prepare
            language: Target language ("en" or "es")
            intents: Optional list of intents to use (cycled if fewer than count)
            sentiment_distribution: Optional custom sentiment distribution
            complexity_distribution: Optional custom complexity distribution
        
        Returns:
            List of prompt dictionaries with keys:
            - prompt: User prompt text
            - system: System prompt text
            - metadata: Dict with intent, sentiment, complexity, language
        """
        prompts = []
        
        # Prepare intent list
        if intents:
            intent_cycle = [intents[i % len(intents)] for i in range(count)]
        else:
            intent_cycle = [None] * count
        
        # Custom distributions
        sent_dist = sentiment_distribution or DEFAULT_DISTRIBUTION["sentiment"]
        comp_dist = complexity_distribution or DEFAULT_DISTRIBUTION["complexity"]
        
        for i in range(count):
            # Select sentiment based on distribution
            sentiments = list(sent_dist.keys())
            weights = list(sent_dist.values())
            selected_sentiment = random.choices(sentiments, weights=weights)[0]
            
            # Select complexity based on distribution
            complexities = list(comp_dist.keys())
            weights = list(comp_dist.values())
            selected_complexity = random.choices(complexities, weights=weights)[0]
            
            # Select parameters
            params = self._select_random_parameters(
                intent=intent_cycle[i],
                sentiment=selected_sentiment,
                complexity=selected_complexity,
                language=language
            )
            
            selected_intent = params["intent"]
            selected_sentiment = params["sentiment"]
            selected_complexity = params["complexity"]
            selected_language = params["language"]
            
            # Convert to enums for prompt builder
            sentiment_enum = Sentiment(selected_sentiment)
            complexity_enum = Complexity(selected_complexity)
            language_enum = Language(selected_language)
            
            # Build prompt with few-shot examples
            prompt_data = build_full_prompt_with_examples(
                intent=selected_intent,
                sentiment=sentiment_enum,
                complexity=complexity_enum,
                language=language_enum,
                num_examples=2,
                emotion_arc=None
            )
            
            prompts.append({
                "prompt": prompt_data["user"],
                "system": prompt_data["system"],
                "metadata": {
                    "intent": selected_intent,
                    "sentiment": selected_sentiment,
                    "complexity": selected_complexity,
                    "language": selected_language,
                    "record_index": i
                }
            })
        
        logger.info(f"Prepared {len(prompts)} prompts for batch inference")
        return prompts
    
    def process_batch_results(
        self,
        results: List[Dict[str, Any]],
        metadata_map: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process results from batch inference into conversation dictionaries.
        
        Args:
            results: Parsed batch results (from BatchResultProcessor)
            metadata_map: Optional mapping of record_id -> metadata
        
        Returns:
            List of validated conversation dictionaries
        """
        conversations = []
        
        for result in results:
            if result.get("error") or not result.get("generated_text"):
                continue
            
            try:
                # Parse generated JSON
                text = result["generated_text"]
                
                # Try to extract JSON from markdown code blocks
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    if end > start:
                        text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    if end > start:
                        text = text[start:end].strip()
                
                conversation = json.loads(text)
                
                # Get metadata for this record
                record_id = result.get("record_id")
                metadata = {}
                if metadata_map and record_id and record_id in metadata_map:
                    metadata = metadata_map[record_id]
                
                # Fix schema issues
                conversation = self._fix_conversation_schema(
                    conversation,
                    intent=metadata.get("intent", "general_inquiry"),
                    sentiment=metadata.get("sentiment", "neutral"),
                    complexity=metadata.get("complexity", "medium"),
                    language=metadata.get("language", "en")
                )
                
                # Add metadata
                conversation["metadata"] = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": self._config.aws.default_model,
                    "generator_version": "1.0.0",
                    "batch_inference": True,
                    "record_id": record_id
                }
                
                # Ensure conversation_id is unique
                if not conversation.get("conversation_id") or conversation["conversation_id"].startswith("conv_XXX"):
                    conversation["conversation_id"] = f"conv_{uuid.uuid4().hex[:12]}"
                
                conversations.append(conversation)
                self._total_generated += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse conversation from record {result.get('record_id')}: {e}")
                self._total_failed += 1
            except Exception as e:
                logger.warning(f"Error processing record {result.get('record_id')}: {e}")
                self._total_failed += 1
        
        logger.info(f"Processed {len(conversations)} conversations from batch results")
        return conversations

