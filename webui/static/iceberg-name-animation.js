(() => {
    "use strict";

    const HOVER_DELAY_MS = 700;
    const STAGGER_MS = 105;
    // Below this scale the unfurled name would be too small to read on one
    // line, so it wraps inside a floating panel instead.
    const MIN_SINGLE_LINE_SCALE = 0.62;
    const WRAP_TARGET_LINES = 3;
    const MIN_WRAP_SCALE = 0.72;
    const COLLAPSE_MS = 680 + STAGGER_MS * 8;

    function initializeIcebergTitle(root) {
        const tails = Array.from(root.querySelectorAll(".iceberg-tail"));
        const isOverlay = root.classList.contains("is-overlay");
        const anchor = root.closest(".iceberg-anchor");
        const header = root.closest("header");
        const bounds = root.closest(".header-inner") || header;
        let hoverTimer = null;
        let wrapResetTimer = null;

        function measureTails() {
            tails.forEach((tail) => {
                tail.style.setProperty("--iceberg-tail-width", `${tail.scrollWidth}px`);
            });
        }

        // Natural width of the fully expanded name on a single line.
        function naturalWidth() {
            return tails.reduce((total, tail) => {
                const initial = tail.previousElementSibling;
                return total + tail.scrollWidth + (initial ? initial.offsetWidth : 0);
            }, 0);
        }

        // Horizontal room between the "I" and the right edge of the header.
        function availableWidth() {
            if (!bounds || !anchor) return Infinity;
            const room = bounds.getBoundingClientRect().right - anchor.getBoundingClientRect().left;
            return Math.max(room - 8, 160);
        }

        function layout() {
            measureTails();
            if (!isOverlay) return;

            const natural = naturalWidth();
            const available = availableWidth();
            if (!natural || !isFinite(available)) return;

            let scale = available / natural;
            let wrap = false;

            if (scale >= 1) {
                scale = 1;
            } else if (scale < MIN_SINGLE_LINE_SCALE) {
                // Not enough room to shrink gracefully: wrap over a few lines.
                wrap = true;
                scale = Math.min(1, Math.max(MIN_WRAP_SCALE, (available * WRAP_TARGET_LINES) / natural));
            }

            root.dataset.wrap = wrap ? "true" : "false";
            root.style.setProperty("--iceberg-scale", scale.toFixed(4));
            root.style.setProperty(
                "--iceberg-wrap-width",
                wrap ? `${Math.round(available / scale)}px` : "none"
            );
        }

        // Keep the in-flow placeholder exactly as wide as the collapsed
        // acronym so the overlay never sits off-register.
        function syncAnchorWidth() {
            if (!anchor || root.dataset.expanded === "true") return;
            anchor.style.minWidth = `${Math.ceil(root.getBoundingClientRect().width)}px`;
        }

        function clearTimers() {
            if (hoverTimer !== null) {
                window.clearTimeout(hoverTimer);
                hoverTimer = null;
            }
            if (wrapResetTimer !== null) {
                window.clearTimeout(wrapResetTimer);
                wrapResetTimer = null;
            }
        }

        // The stagger lives on the word so the tail and the initial's
        // underline retract/return in lockstep.
        function setDelays(expanding) {
            tails.forEach((tail, index) => {
                const word = tail.parentElement || tail;
                const order = expanding ? index : tails.length - 1 - index;
                word.style.setProperty("--iceberg-delay", `${order * STAGGER_MS}ms`);
            });
        }

        function expand() {
            clearTimers();
            layout();
            setDelays(true);
            root.dataset.expanded = "true";
            if (header) header.dataset.icebergExpanded = "true";
        }

        function scheduleExpansion() {
            clearTimers();
            layout();
            hoverTimer = window.setTimeout(() => {
                hoverTimer = null;
                expand();
            }, HOVER_DELAY_MS);
        }

        function collapse() {
            clearTimers();
            setDelays(false);
            root.dataset.expanded = "false";
            if (header) delete header.dataset.icebergExpanded;
            root.style.setProperty("--iceberg-scale", "1");
            // Keep the wrapped geometry until the tails have retracted, so the
            // panel does not snap back to a single line mid-animation.
            if (root.dataset.wrap === "true") {
                wrapResetTimer = window.setTimeout(() => {
                    wrapResetTimer = null;
                    root.dataset.wrap = "false";
                    root.style.setProperty("--iceberg-wrap-width", "none");
                }, COLLAPSE_MS);
            }
        }

        const resizeObserver = new ResizeObserver(() => {
            if (root.dataset.expanded === "true") {
                layout();
            } else {
                measureTails();
            }
        });
        if (bounds) resizeObserver.observe(bounds);

        // Hover/leave is bound to the animated element itself so that moving
        // the pointer across the unfurled name keeps it open.
        root.addEventListener("pointerenter", scheduleExpansion);
        root.addEventListener("pointerleave", collapse);

        if (anchor) {
            // Keyboard + touch: reveal immediately, no dwell delay.
            anchor.addEventListener("focus", expand);
            anchor.addEventListener("blur", collapse);
            anchor.addEventListener("keydown", (event) => {
                if (event.key === "Escape") collapse();
            });
        }

        collapse();
        layout();
        requestAnimationFrame(syncAnchorWidth);
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(() => {
                syncAnchorWidth();
                layout();
            });
        }
    }

    function initialize() {
        document.querySelectorAll(".iceberg-title-animation").forEach(initializeIcebergTitle);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})();
