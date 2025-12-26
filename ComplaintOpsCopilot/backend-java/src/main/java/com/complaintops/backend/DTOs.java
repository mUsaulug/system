package com.complaintops.backend;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class DTOs {

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class MaskingRequest {
        private String text;
    }

    @Data
    public static class MaskingResponse {
        @JsonProperty("original_text")
        private String originalText;

        @JsonProperty("masked_text")
        private String maskedText;

        @JsonProperty("masked_entities")
        private List<String> maskedEntities;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class TriageRequest {
        private String text;
    }

    @Data
    public static class TriageResponse {
        private String category;

        @JsonProperty("category_confidence")
        private double categoryConfidence;

        private String urgency;

        @JsonProperty("urgency_confidence")
        private double urgencyConfidence;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class RAGRequest {
        private String text;
    }

    @Data
    public static class RAGResponse {
        @JsonProperty("relevant_snippets")
        private List<String> relevantSnippets;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class GenerateRequest {
        private String text;
        private String category;
        private String urgency;

        @JsonProperty("relevant_snippets")
        private List<String> relevantSnippets;
    }

    @Data
    public static class GenerateResponse {
        @JsonProperty("action_plan")
        private List<String> actionPlan;

        @JsonProperty("customer_reply_draft")
        private String customerReplyDraft;

        @JsonProperty("risk_flags")
        private List<String> riskFlags;
    }
}
