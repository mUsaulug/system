package com.complaintops.backend;

import org.springframework.web.bind.annotation.*;
import lombok.RequiredArgsConstructor;
import lombok.Data;
import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "*") // Allow all for MVP
public class ComplaintController {

    private final OrchestratorService orchestratorService;

    @GetMapping("/complaints")
    public List<Complaint> getAllComplaints() {
        return orchestratorService.getAllComplaints();
    }

    @GetMapping("/complaints/{id}")
    public Complaint getComplaint(@PathVariable Long id) {
        return orchestratorService.getComplaint(id);
    }

    @PostMapping("/analyze")
    public Complaint analyzeComplaint(@RequestBody ComplaintRequest request) {
        return orchestratorService.analyzeComplaint(request.getText());
    }

    @Data
    static class ComplaintRequest {
        private String text;
    }
}
