# Implementation Plan: Now → May 13, 2026
**Graph Console & Tree View System**

---

## Overview

We have two hard milestones:

| Date | Event |
|------|-------|
| **May 6 (Tuesday)** | On-site visit — demo/review of mapping + form generation phase |
| **May 13 (Wednesday)** | Final submission deadline |

The plan below is split into two phases accordingly.

---

## Phase 1: Pre On-Site (Now → May 5)
**Goal: Have a working demo of API Mapping + Form Generation pipeline ready for on-site review on May 6.**

---

### Day 1–2 (May 1–2): Config JSON & Static Graph Backbone

**Deliverable: Static Config JSON skeleton + graph backbone defined**

- [ ] Define the core nomenclature structure manually:
  - PFM (root)
  - WEA, THR, MOD (L1 entities)
  - Sub-levels with placeholder Group IDs, Node IDs
- [ ] Create the static Config JSON file with the following schema per node:
  ```json
  {
    "node_id": "WEA-001",
    "label": "WEA",
    "type": "WEA",
    "level": 1,
    "parent_id": "PFM-000",
    "group_id": "GRP-WEA-01",
    "is_leaf": false,
    "children": []
  }
  ```
- [ ] Ensure the JSON covers all levels (L0 → L3 at minimum) for WEA, THR, MOD branches
- [ ] Validate the JSON structure — no orphan nodes, all parent IDs resolve

---

### Day 3–4 (May 3–4): API Mapping Script + Form Template Store

**Deliverable: Working mapping script that enriches the OpenAPI JSON and links forms to graph nodes**

- [ ] Write the **mapping script** that:
  - Takes the existing OpenAPI JSON (~512 APIs)
  - Injects `unit`, `type`, `format`, `range` fields at the property level
  - Wraps field definitions with `session_id` and `group_id` global variables
  - Outputs an enriched API JSON
- [ ] Write the **matching utility** that:
  - Takes the enriched API JSON
  - Maps each API form to a specific Node ID in the static Config JSON
  - Outputs the unified Config JSON (single source of truth)
- [ ] Test with at least 10–15 API entries end-to-end

---

### Day 5 (May 5): On-Site Demo Prep

**Deliverable: Clean demo-ready build**

- [ ] Run the full pipeline end-to-end on a sample set of APIs
- [ ] Prepare a short walkthrough script for the on-site demo:
  1. Show the raw OpenAPI JSON
  2. Run the mapping script → show enriched output
  3. Run the matching utility → show unified Config JSON
  4. Show the static graph backbone rendered (even as a simple tree print or basic UI stub)
- [ ] Confirm that the encrypted Config JSON output is readable and correct
- [ ] Prepare any questions for the on-site meeting regarding mapping variables from the encoder

---

## May 6: On-Site Visit
**What to demo:**
- The mapping + form generation pipeline (Phase 1 output)
- The static Config JSON backbone
- A basic visual stub of the graph structure (even hand-drawn or low-fidelity is fine)

**What to collect:**
- Mapping variables and encoder details from the reviewer
- Any additional nomenclature or node naming conventions
- Confirm the exact encrypted JSON format expected

---

## Phase 2: Post On-Site → Final Submission (May 7–13)
**Goal: Build the full Graph UI + Main UI with all features and submit by May 13.**

---

### May 7–8: Graph UI — Core Shell

**Deliverable: Basic left-panel tree + right-panel graph shell (no interaction yet)**

- [ ] Set up the React project (if not already done)
- [ ] Integrate D3.js or React Flow for the graph canvas
- [ ] Build the left-panel collapsible tree component:
  - Load from Config JSON
  - Expand/collapse per node
  - Session persistence (remember last state)
- [ ] Render the right-panel graph from the same Config JSON
- [ ] Implement tab switching: PFM | WEA | THR | MOD
- [ ] Visually distinguish node types with shapes + colours:
  - WEA: shape TBD, colour A
  - THR: shape TBD, colour B
  - MOD: shape TBD, colour C
  - ST: rectangle, high-contrast colour (top level)

---

### May 9–10: Graph UI — Interactions

**Deliverable: All node operations + CQL + drag-and-drop working**

- [ ] Implement node operations (right-click / context menu):
  - Add Child Node
  - Delete Node
  - Modify Node
  - Modify Access
  - Modify View
- [ ] Implement CQL (Custom Query Language) input:
  - Input field for: `GROUP_ID + NODE_ID`
  - On submit: navigate to / highlight that node in the graph and tree
  - On insert command: add a node at the specified position
- [ ] Implement drag-and-drop from left-panel tree to graph canvas
- [ ] Ensure tree ↔ graph synchronisation in real-time (any change in one reflects in the other)
- [ ] Implement the Logs & History footer panel (timestamped audit log)

---

### May 11: Main UI Integration

**Deliverable: All three subsystems connected into one unified UI**

- [ ] Wire the API Mapper output → Config JSON → Graph UI
- [ ] Implement the encrypted Config JSON read/write flow
- [ ] Ensure form definitions load correctly when a leaf node is selected
- [ ] Handle session ID and group ID injection from the active application state
- [ ] Test the full end-to-end flow: API → Form → Graph Node → Config JSON

---

### May 12: Testing & Polish

- [ ] Full regression test of all node operations
- [ ] Test tab switching and root node changes
- [ ] Verify left-panel session memory persists across reloads
- [ ] Check all visual indicators (colours, shapes, ST rectangle) are correct
- [ ] Edge cases: orphan nodes, empty graph start state, max-depth limits

---

### May 13: Submission

- [ ] Final build + deployment
- [ ] Submit the following artefacts:
  - Unified Config JSON (encrypted)
  - Mapping + enrichment scripts
  - Graph UI + Main UI codebase
  - Documentation (this plan + UI spec)

---

## Summary Gantt

```
May 1  ██ Config JSON backbone
May 2  ██ Config JSON backbone
May 3  ██ Mapping script + form template store
May 4  ██ Matching utility + end-to-end test
May 5  ██ Demo prep
May 6  🏢 ON-SITE VISIT
May 7  
May 8  
May 9  ██ Node operations + CQL
May 10 
May 12 ██ Testing & polish
May 13 🚀 SUBMISSION
```
