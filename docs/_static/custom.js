(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function currentPath() {
    try {
      return decodeURIComponent(window.location.pathname);
    } catch {
      return window.location.pathname;
    }
  }

  function markHome() {
    const path = currentPath();
    if (
      /\/(index\.html)?$/.test(path) ||
      path.endsWith("/NetForge_RL") ||
      path.endsWith("/NetForge_RL/")
    ) {
      document.documentElement.classList.add("fe-home");
    }
  }

  function markNav() {
    document.querySelectorAll(".fe-topnav-link").forEach((a) => {
      try {
        const href = new URL(a.href, window.location.href);
        if (href.pathname === window.location.pathname) a.classList.add("is-current");
      } catch {
        /* ignore */
      }
    });
  }

  function reveal() {
    const nodes = document.querySelectorAll(
      ".fe-news-item, .fe-fig, .fe-card, article table, .fe-explorer"
    );
    if (reduceMotion || !("IntersectionObserver" in window)) {
      nodes.forEach((el) => el.classList.add("is-in"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -6% 0px", threshold: 0.05 }
    );
    nodes.forEach((el) => {
      el.classList.add("fe-reveal");
      const top = el.getBoundingClientRect().top;
      if (top < window.innerHeight * 0.94) {
        el.classList.add("is-in");
      } else {
        io.observe(el);
      }
    });
  }

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  ready(() => {
    markHome();
    markNav();
    reveal();
  });
})();
