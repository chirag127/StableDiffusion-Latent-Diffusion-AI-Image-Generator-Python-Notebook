# 🚀 Pull Request Submission Template

This template ensures that all contributions adhere to the **Apex Technical Authority** standards for maintainability, performance, and clarity. Review this checklist before submitting.

---

## 🎯 Summary of Changes

**Briefly describe the feature, fix, or refactor in 1-2 sentences.**

<!-- e.g., Fixes an off-by-one error in the U-Net forward pass, or Refactors VAE decoder for better memory locality. -->

---

## 📝 Detailed Context

Provide comprehensive detail on *why* these changes were necessary and *how* they were implemented.

### Motivation
<!-- Why is this change needed? Link to any relevant issues (e.g., Closes #123). -->

### Implementation Details
<!-- Describe architectural changes, new dependencies, or complex logic involved. Reference specific files if necessary. -->

---

## ✅ Verification Checklist (Self-Review)

*Ensure all applicable points are checked before requesting review.*

- [ ] **Code Style:** Formatted code using **Ruff** (or appropriate Python formatter). 
- [ ] **Documentation:** Updated relevant docstrings/comments where logic changed significantly.
- [ ] **Testing:** Added or updated **Pytest** unit/integration tests that cover the new/modified logic.
- [ ] **Architecture Adherence:** Changes respect the **Modular Monolith** principles and **SOLID** design patterns established in the project.
- [ ] **Performance:** Verified no significant regressions (especially relevant for PyTorch models).
- [ ] **Security:** No new vulnerabilities introduced (especially concerning deserialization or external data handling).

---

## 🧪 Testing Strategy

Describe *how* you locally tested these changes. For Stable Diffusion, specific input/output examples are highly valuable.

1.  **Test Case 1:** (e.g., Ran `pytest tests/unit/test_unet.py`)
2.  **Test Case 2:** (e.g., Synthesized image using `--prompt 'An astronaut riding a horse' --steps 20`)

---

## 📸 Screenshots / Artifacts (If Applicable)

Paste any relevant generated images, performance benchmarks, or CLI output here.

---

## 👁️ Reviewer Guidance

Direct the reviewer to the most critical areas needing attention (e.g., "Please focus review on `src/scheduler.py` lines 45-60, as this involves new integration with `torch.compile`.").
