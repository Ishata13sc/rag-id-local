function toast(msg) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.remove("hidden");
  t.style.opacity = "1";
  setTimeout(() => {
    t.style.opacity = "0";
  }, 1800);
  setTimeout(() => {
    t.classList.add("hidden");
  }, 2200);
}
function openModal() {
  document.getElementById("modal").classList.remove("hidden");
}
function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

document.addEventListener("htmx:afterRequest", (e) => {
  if (e.detail.successful && e.detail.path) {
    if (e.detail.path.includes("/upload")) toast("Uploaded");
    if (e.detail.path.includes("/ingest")) toast("Ingested");
    if (e.detail.path.includes("/search")) toast("Search done");
    if (e.detail.path.includes("/admin/reindex")) toast("Index rebuilt");
  }
});

document.addEventListener("htmx:afterSwap", (e) => {
  if (e.detail.target && e.detail.target.id === "ingest-list") {
    const host = e.detail.target;
    const items = host.querySelectorAll("[data-upload-name]");
    items.forEach((card) => {
      const name = card.getAttribute("data-upload-name");
      const btn = card.querySelector('button[data-action="ingest"]');
      if (btn && name) {
        btn.setAttribute(
          "hx-get",
          `/ingest?filename=${encodeURIComponent(name)}`
        );
        btn.setAttribute("hx-target", "#ingest-result");
        btn.setAttribute("hx-swap", "innerHTML");
      }
    });
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});
