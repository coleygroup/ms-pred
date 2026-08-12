(() => {
    "use strict";

    const HOVER_DELAY_MS = 1000;
    const STAGGER_MS = 105;

    function initializeIcebergTitle(root) {
        const tails = Array.from(root.querySelectorAll(".iceberg-tail"));
        let hoverTimer = null;

        function measure() {
            tails.forEach((tail) => {
                tail.style.setProperty("--iceberg-tail-width", `${tail.scrollWidth}px`);
            });
        }

        function clearHoverTimer() {
            if (hoverTimer !== null) {
                window.clearTimeout(hoverTimer);
                hoverTimer = null;
            }
        }

        function setDelays(expanding) {
            tails.forEach((tail, index) => {
                const order = expanding ? index : tails.length - 1 - index;
                tail.style.transitionDelay = `${order * STAGGER_MS}ms`;
            });
        }

        function scheduleExpansion() {
            clearHoverTimer();
            measure();
            hoverTimer = window.setTimeout(() => {
                setDelays(true);
                root.dataset.expanded = "true";
                hoverTimer = null;
            }, HOVER_DELAY_MS);
        }

        function collapse() {
            clearHoverTimer();
            setDelays(false);
            root.dataset.expanded = "false";
        }

        const resizeObserver = new ResizeObserver(measure);
        resizeObserver.observe(root);
        root.addEventListener("pointerenter", scheduleExpansion);
        root.addEventListener("pointerleave", collapse);
        measure();
        collapse();
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
