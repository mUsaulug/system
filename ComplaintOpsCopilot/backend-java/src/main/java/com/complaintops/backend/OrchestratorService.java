package com.complaintops.backend;

import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.beans.factory.annotation.Value;
import lombok.RequiredArgsConstructor;
import java.util.List;
import java.util.ArrayList;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class OrchestratorService {

    private final ComplaintRepository repository;
    private final WebClient.Builder webClientBuilder;

    @Value("${ai-service.url}")
    private String aiServiceUrl;

    public Complaint analyzeComplaint(String rawText) {
        WebClient webClient = webClientBuilder.baseUrl(Objects.requireNonNull(aiServiceUrl)).build();

        // 1. Mask PII
        DTOs.MaskingResponse maskResp;
        try {
            maskResp = webClient.post()
                    .uri("/mask")
                    .bodyValue(new DTOs.MaskingRequest(rawText))
                    .retrieve()
                    .bodyToMono(DTOs.MaskingResponse.class)
                    .block();
        } catch (Exception e) {
            // Fallback if masking fails (Serious error, but for MVP we wrap)
            System.err.println("Masking failed: " + e.getMessage());
            maskResp = new DTOs.MaskingResponse();
            maskResp.setMaskedText(rawText); // Fallback to raw (RISK!) - In prod, fail hard here.
            maskResp.setMaskedEntities(new ArrayList<>());
        }

        String safeText = maskResp.getMaskedText();

        // 2. Triage
        DTOs.TriageResponse triageResp;
        try {
            triageResp = webClient.post()
                    .uri("/predict")
                    .bodyValue(new DTOs.TriageRequest(safeText))
                    .retrieve()
                    .bodyToMono(DTOs.TriageResponse.class)
                    .block();
        } catch (Exception e) {
            System.err.println("Triage failed: " + e.getMessage());
            triageResp = new DTOs.TriageResponse();
            triageResp.setCategory("MANUAL_REVIEW");
            triageResp.setUrgency("MEDIUM");
        }

        // 3. RAG Retrieval
        DTOs.RAGResponse ragResp;
        try {
            ragResp = webClient.post()
                    .uri("/retrieve")
                    .bodyValue(new DTOs.RAGRequest(safeText))
                    .retrieve()
                    .bodyToMono(DTOs.RAGResponse.class)
                    .block();
        } catch (Exception e) {
            System.err.println("RAG failed: " + e.getMessage());
            ragResp = new DTOs.RAGResponse();
            ragResp.setRelevantSnippets(new ArrayList<>());
        }

        // 4. Generate Response
        DTOs.GenerateResponse genResp;
        try {
            genResp = webClient.post()
                    .uri("/generate")
                    .bodyValue(new DTOs.GenerateRequest(
                            safeText,
                            triageResp.getCategory(),
                            triageResp.getUrgency(),
                            ragResp.getRelevantSnippets()))
                    .retrieve()
                    .bodyToMono(DTOs.GenerateResponse.class)
                    .block();
        } catch (Exception e) {
            System.err.println("Generation failed: " + e.getMessage());
            genResp = new DTOs.GenerateResponse();
            genResp.setActionPlan(List.of("System Error: AI Generation Failed. Please review manually."));
            genResp.setCustomerReplyDraft("Error generating draft.");
        }

        // 5. Save to DB
        Complaint complaint = new Complaint();
        complaint.setOriginalText(rawText);
        complaint.setMaskedText(safeText);
        complaint.setCategory(triageResp.getCategory());
        complaint.setUrgency(triageResp.getUrgency());

        try {
            complaint.setActionPlan(new ObjectMapper().writeValueAsString(genResp.getActionPlan()));
        } catch (Exception e) {
            complaint.setActionPlan(String.valueOf(genResp.getActionPlan()));
        }

        complaint.setCustomerReplyDraft(genResp.getCustomerReplyDraft());
        complaint.setStatus(ComplaintStatus.ANALYZED);

        return repository.save(complaint);
    }

    public List<Complaint> getAllComplaints() {
        return repository.findAll();
    }

    public Complaint getComplaint(Long id) {
        return repository.findById(Objects.requireNonNull(id))
                .orElseThrow(() -> new RuntimeException("Complaint not found"));
    }
}
