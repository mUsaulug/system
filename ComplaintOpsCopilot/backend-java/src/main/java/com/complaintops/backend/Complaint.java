package com.complaintops.backend;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(name = "complaints")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Complaint {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(columnDefinition = "TEXT")
    private String originalText;

    @Column(columnDefinition = "TEXT")
    private String maskedText;

    private String category;
    private String urgency;

    @Column(columnDefinition = "TEXT")
    private String actionPlan; // Stored as JSON string or simple text

    @Column(columnDefinition = "TEXT")
    private String customerReplyDraft;

    @Enumerated(EnumType.STRING)
    private ComplaintStatus status = ComplaintStatus.NEW;

    private LocalDateTime createdAt = LocalDateTime.now();
}

enum ComplaintStatus {
    NEW,
    ANALYZED,
    RESOLVED
}
