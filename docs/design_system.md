# MedInspect interface design system

The interface uses a quiet clinical workspace: a dark navigation rail, white task surfaces, a neutral background, and a single teal accent. Native Streamlit controls retain keyboard navigation, labels, and screen-reader behavior. Presentation helpers live in `src/ui.py`; reusable tokens and component styles live in `assets/styles.css`.

## Foundation

| Rule | Values |
| --- | --- |
| Font | One variable sans serif, Source Sans, served from the pinned Streamlit package. No external font request. |
| Page heading | 32 px, weight 650. 28 px on mobile. |
| Section heading | 18 px, weight 650. |
| Body and input | 16 px. |
| Labels and supporting text | 14 px labels/help; 12 px compact metadata. |
| Spacing | Multiples of 4 px. Primary steps: 8, 12, 16, 24, and 32 px. |
| Icons | Material Symbols Rounded throughout, including native widget icons. Usually 20 px; supporting icons 16 px. Decorative icons are hidden from assistive technology. |
| Accent | `#0F766E`, hover `#0B615A`, pale accent `#EDF7F4`. |
| Surfaces | Canvas `#F6F8F9`, surface white, rail `#142E35`. |
| Text | Primary `#1B3039`, secondary `#536873`. |
| Borders | `#DFE6E9`, with 8 px control corners and 12 px surface corners. |

## Component and interaction rules

One inspection surface groups instrument details, image upload, and the primary action. The progress indicator reflects actual completed inputs, inference, and a saved operator review. A compact session strip shows counts without competing with the task. Result and review surfaces share the same border, corner, heading, and padding rules. Empty states stay compact. The sidebar holds status and expandable guidance rather than duplicating the workflow.

The primary action has a consistent filled teal treatment, secondary actions use white surfaces, and disabled actions include a visible explanation. Search and decision filters change only the displayed records; full-session CSV export remains available. After saving, the next-inspection action preserves the batch and report. All interactive controls retain native labels and keyboard behavior.

Model connection status belongs in the rail. Missing-model setup details are expandable, while the main action states why analysis is unavailable. No synthetic statistics, simulated predictions, invented user identities, or unsupported controls appear in the interface. The research limitation remains visible below the workspace.

On small screens, task columns stack, spacing reduces, and typography preserves its hierarchy. Focus outlines remain visible. Reduced-motion preferences disable decorative transitions. Long IDs and model names wrap rather than expanding the layout.

The stylesheet scopes panel styling to documented Streamlit container keys and uses native component selectors for controls. Recheck browser screenshots and the existing workflow tests when upgrading Streamlit, because those component selectors may change.
