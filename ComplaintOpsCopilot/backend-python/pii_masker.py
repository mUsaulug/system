from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import List, Dict

class PIIMasker:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.pdf_analyzer = None # Placeholder for PDF analysis if needed
        
        # Add Custom Recognizer for Turkish TCKN (Identity Number)
        # TCKN is 11 digits, valid algorithm check is complex but for regex we can use \d{11}
        # and maybe context words like "TC", "TCKN", "Kimlik"
        tckn_pattern = Pattern(name="tckn_pattern", regex=r"\b[1-9][0-9]{10}\b", score=0.5)
        tckn_recognizer = PatternRecognizer(
            supported_entity="TCKN",
            patterns=[tckn_pattern],
            context=["tc", "tckn", "kimlik", "no", "numarası"]
        )
        self.analyzer.registry.add_recognizer(tckn_recognizer)

        # IBAN is usually supported, but we can verify or add specific TR IBAN regex
        # TR IBAN: TR + 24 digits
        tr_iban_pattern = Pattern(name="tr_iban_pattern", regex=r"TR\d{2}\s?(\d{4}\s?){6}", score=0.8)
        tr_iban_recognizer = PatternRecognizer(
            supported_entity="TR_IBAN",
            patterns=[tr_iban_pattern],
            context=["iban", "hesap"]
        )
        self.analyzer.registry.add_recognizer(tr_iban_recognizer)

    def mask(self, text: str) -> Dict:
        # Analyze
        results = self.analyzer.analyze(text=text, entities=["TCKN", "TR_IBAN", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"], language='en') # 'en' model is often good enough for numbers/regex, 'tr' support depends on spacy model installed
        
        # Anonymize
        # We want to replace with [MASKED_ENTITY_TYPE]
        operators = {
            "TCKN": OperatorConfig("replace", {"new_value": "[MASKED_TCKN]"}),
            "TR_IBAN": OperatorConfig("replace", {"new_value": "[MASKED_IBAN]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[MASKED_PHONE]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[MASKED_EMAIL]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[MASKED_CC]"}),
        }
        
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        
        return {
            "original_text": text,
            "masked_text": anonymized_result.text,
            "masked_entities": [res.entity_type for res in results]
        }

# Global instance
masker = PIIMasker()
