import os
import streamlit as st
from config import load_config, save_config, CONFIG_PATH

st.set_page_config(page_title="Hateful Memes Pipeline", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()

def _render_log(entries: list, container) -> None:
    # container ist ein st.empty()-Platzhalter: .markdown() ersetzt den Inhalt,
    # statt anzuhängen (sonst akkumulieren sich die Logs 1+2+3+...).
    container.markdown("\n".join(entries[:50]))

tab_s, tab_p1, tab_p2, tab_r = st.tabs(
    ["⚙️ Einstellungen", "📷 Phase 1", "🧠 Phase 2", "🚀 Experiment-Runner"])

# ── EINSTELLUNGEN ─────────────────────────────────────────────────────────────
with tab_s:
    st.header("Einstellungen")
    cfg = st.session_state.cfg

    cfg["prompt_excel"]   = st.text_input("Prompt-Excel", value=cfg.get("prompt_excel",""))
    cfg["img_folder"]     = st.text_input("Bild-Ordner (img/)", value=cfg.get("img_folder",""))
    cfg["results_folder"] = st.text_input("Ergebnis-Ordner", value=cfg.get("results_folder",""))

    c1, c2 = st.columns(2)
    with c1:
        cfg["max_tokens_phase1"] = st.number_input(
            "Max. Token pro Bildbeschreibung", 100, 10000,
            value=cfg.get("max_tokens_phase1", 2500), step=100)
    with c2:
        cfg["max_time_seconds"] = st.number_input(
            "Max. Zeit pro Bild (Sek.)", 10, 600,
            value=cfg.get("max_time_seconds", 120), step=10)

    cfg["phase1_excel"]  = st.text_input("Phase-1-Ergebnis-Excel", value=cfg.get("phase1_excel",""))
    cfg["phase2_excel"]  = st.text_input("Phase-2-Ergebnis-Excel", value=cfg.get("phase2_excel",""))
    cfg["ft_model_path"] = st.text_input(
        "Fine-Tuned Modell (leer lassen bis FT fertig)",
        value=cfg.get("ft_model_path",""))

    if st.button("💾 Speichern"):
        save_config(CONFIG_PATH, cfg)
        st.session_state.cfg = cfg
        st.success("Gespeichert.")

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
with tab_p1:
    st.header("Phase 1 — Bildbeschreibungen")
    cfg = st.session_state.cfg

    prompts_p1 = {}
    if cfg.get("prompt_excel") and os.path.exists(cfg["prompt_excel"]):
        from excel_utils import read_prompts
        try:
            prompts_p1 = read_prompts(cfg["prompt_excel"], "Phase1")
        except Exception as e:
            st.warning(f"Prompt-Excel: {e}")

    if not prompts_p1:
        st.info("Bitte Prompt-Excel in Einstellungen hinterlegen.")
    else:
        mode_p1 = st.radio("Modus",
            ["Experiment-Modus (dev.jsonl)", "Fine-Tuning-Modus (train.jsonl)"],
            horizontal=True,
            help="Experiment: ~500 Bilder für Phase-2-Auswertung. FT: 8500 Bilder für QLoRA.")
        jsonl_name = "dev.jsonl" if "Experiment" in mode_p1 else "train.jsonl"
        jsonl_path = os.path.join(os.path.dirname(cfg.get("img_folder","")), jsonl_name)

        prompt_name = st.selectbox("Prompt", list(prompts_p1.keys()))
        prompt_text = st.text_area("Prompt-Text", value=prompts_p1[prompt_name], height=150)

        from phase1 import get_run_info, clear_run
        run_info = get_run_info(prompt_name, cfg.get("results_folder",""))
        resume = False
        if run_info:
            ca, cb = st.columns([3, 1])
            with ca:
                st.info(f"Unterbrochener Lauf: {run_info['n_processed']} Einträge bereits verarbeitet")
            with cb:
                resume = st.checkbox("Fortsetzen", value=True)
                if st.button("Neu starten"):
                    clear_run(prompt_name, cfg["results_folder"])
                    st.rerun()

        if st.button("▶️ Phase 1 starten"):
            if not cfg.get("img_folder") or not cfg.get("phase1_excel"):
                st.error("Bitte Bild-Ordner und Phase-1-Excel angeben.")
            elif not os.path.exists(jsonl_path):
                st.error(f"{jsonl_name} nicht gefunden: {jsonl_path}")
            else:
                from phase1 import run_phase1
                bar = st.progress(0.0)
                log_box = st.empty()
                log_entries = []
                for upd in run_phase1(
                    jsonl_path=jsonl_path, img_folder=cfg["img_folder"],
                    phase1_excel=cfg["phase1_excel"], prompt_name=prompt_name,
                    prompt_text=prompt_text, max_tokens=cfg["max_tokens_phase1"],
                    max_time_secs=cfg["max_time_seconds"],
                    results_folder=cfg["results_folder"], resume=resume):
                    if upd["type"] == "progress":
                        bar.progress(upd["current"] / max(upd["total"], 1),
                                     text=f"{upd['current']} / {upd['total']}")
                    elif upd["type"] == "log":
                        icon = "✅" if upd["status"] == "ok" else "⚠️"
                        log_entries.insert(0,
                            f"{icon} **ID {upd['id']}** | _{upd['text'][:60]}_\n\n"
                            f"{upd['description']}\n\n---")
                        _render_log(log_entries, log_box)
                    elif upd["type"] == "done":
                        st.success(f"Fertig! ✅ {upd['total_ok']} OK, ⚠️ {upd['total_skip']} übersprungen")

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
with tab_p2:
    st.header("Phase 2 — Klassifizierung")
    cfg = st.session_state.cfg

    prompts_p2 = {}
    if cfg.get("prompt_excel") and os.path.exists(cfg["prompt_excel"]):
        from excel_utils import read_prompts
        try:
            prompts_p2 = read_prompts(cfg["prompt_excel"], "Phase2")
        except Exception as e:
            st.warning(f"Prompt-Excel: {e}")

    p1_sheets = []
    if cfg.get("phase1_excel") and os.path.exists(cfg["phase1_excel"]):
        from excel_utils import get_sheet_names
        p1_sheets = [s for s in get_sheet_names(cfg["phase1_excel"])
                     if not s.startswith("M_")]

    if not p1_sheets:
        st.info("Zuerst Phase 1 im Experiment-Modus ausführen.")
    elif not prompts_p2:
        st.info("Prompt-Excel in Einstellungen hinterlegen.")
    else:
        p1_sheet   = st.selectbox("Phase-1-Sheet", p1_sheets)
        p_name     = st.selectbox("Basis-Prompt", list(prompts_p2.keys()))
        p_text     = st.text_area("Prompt-Text", value=prompts_p2.get(p_name,""), height=200)
        use_rag    = st.checkbox("RAG aktivieren")
        use_ft     = st.checkbox("Fine-Tuned Modell",
                                 disabled=not bool(cfg.get("ft_model_path")),
                                 help="FT-Pfad in Einstellungen eintragen.")

        if st.button("▶️ Single Run"):
            dev_jsonl = os.path.join(os.path.dirname(cfg["img_folder"]), "dev.jsonl")
            rag_r = None
            if use_rag:
                from rag import RagRetriever
                rag_r = RagRetriever(os.path.join(cfg["results_folder"], "chroma_db"))
            bar = st.progress(0.0)
            log_box = st.empty()
            log_entries = []
            from phase2 import run_phase2
            for upd in run_phase2(
                phase1_excel=cfg["phase1_excel"], phase1_sheet=p1_sheet,
                dev_jsonl_path=dev_jsonl, phase2_excel=cfg["phase2_excel"],
                prompt_name=p_name, prompt_text=p_text,
                use_rag=use_rag, use_ft=use_ft,
                results_folder=cfg["results_folder"],
                ft_model_path=cfg.get("ft_model_path",""),
                rag_retriever=rag_r):
                if upd["type"] == "progress":
                    bar.progress(upd["current"] / max(upd["total"], 1),
                                 text=f"{upd['current']} / {upd['total']}")
                elif upd["type"] == "log":
                    icon = "🔴" if upd["label"] == 1 else "🟢"
                    log_entries.insert(0,
                        f"{icon} **ID {upd['id']}** | Label: {upd['label']} | Conf: {upd['confidence']:.0f}%\n\n"
                        f"_{upd['text'][:80]}_\n\n**Reasoning:** {upd['reasoning']}\n\n---")
                    _render_log(log_entries, log_box)
                elif upd["type"] == "done":
                    m = upd.get("metrics", {})
                    if m:
                        st.success(f"✅ Acc {m.get('accuracy',0):.1%} | "
                                   f"AUROC {m.get('auroc','n/a')} | "
                                   f"F1 {m.get('f1',0):.3f} | n={m.get('n_samples',0)}")

# ── EXPERIMENT-RUNNER ─────────────────────────────────────────────────────────
with tab_r:
    st.header("Experiment-Runner")
    cfg = st.session_state.cfg

    st.subheader("Phase 1")
    prompts_r1 = {}
    if cfg.get("prompt_excel"):
        from excel_utils import read_prompts
        try:
            prompts_r1 = read_prompts(cfg["prompt_excel"], "Phase1")
        except Exception:
            pass
    p1_opts = list(prompts_r1.keys()) if prompts_r1 else ["ZS", "ZS+RP+AD", "ZS+RP+CoT+AD", "ZS+RP+AD(min)"]
    p1_sel = [n for n in p1_opts if st.checkbox(n, key=f"r1_{n}")]

    st.subheader("Phase 2")
    st.caption("Optional — wenn leer wird nur Phase 1 durchgeführt.")
    p2_opts = ["ZS", "ZS+RAG", "AD+RP+RAG", "AD+RP+CoT+RAG", "ZS+FT"]
    p2_sel = [n for n in p2_opts if st.checkbox(n, key=f"r2_{n}")]

    n_p1 = len(p1_sel)
    n_combos = len(p1_sel) * len(p2_sel)
    if p2_sel:
        st.info(f"**{n_p1} Phase-1-Prompt(s)** → **{n_combos} Phase-2-Kombinationen**")
    else:
        st.info(f"**{n_p1} Phase-1-Prompt(s)** ausgewählt (nur Phase 1)")

    if st.button("🚀 Starten", disabled=(n_p1 == 0)):
        prompts_r2 = {}
        if cfg.get("prompt_excel"):
            from excel_utils import read_prompts
            try:
                prompts_r2 = read_prompts(cfg["prompt_excel"], "Phase2")
            except Exception:
                pass
        rag_r = None
        if any("RAG" in p for p in p2_sel):
            from rag import RagRetriever
            rag_r = RagRetriever(os.path.join(cfg["results_folder"], "chroma_db"))
        dev_jsonl = os.path.join(os.path.dirname(cfg["img_folder"]), "dev.jsonl")
        phase1_info = st.empty()
        combo_info  = st.empty()
        bar_total   = st.progress(0.0)
        bar_combo   = st.progress(0.0)
        log_box     = st.empty()
        log_entries = []
        n_done = 0
        from experiment_runner import run_experiments
        for upd in run_experiments(
            phase1_selections=p1_sel, phase2_selections=p2_sel,
            prompts_phase1=prompts_r1, prompts_phase2=prompts_r2,
            phase1_excel=cfg["phase1_excel"],
            dev_jsonl_path=dev_jsonl, phase2_excel=cfg["phase2_excel"],
            results_folder=cfg["results_folder"],
            img_folder=cfg["img_folder"],
            max_tokens=cfg.get("max_tokens_phase1", 3000),
            max_time_secs=cfg.get("max_time_seconds", 150),
            ft_model_path=cfg.get("ft_model_path",""), rag_retriever=rag_r):
            if upd["type"] == "phase1_start":
                phase1_info.info(f"⏳ Phase 1: **{upd['prompt']}** "
                                 f"({upd['index']}/{upd['total']})")
                bar_total.progress(0.0)
                bar_combo.progress(0.0)
            elif upd["type"] == "phase1_skip":
                phase1_info.info(f"⏭️ Phase 1: **{upd['prompt']}** bereits vorhanden — übersprungen")
            elif upd["type"] == "phase1_done":
                phase1_info.success(f"✅ Phase 1: **{upd['prompt']}** abgeschlossen")
            elif upd["type"] == "combination_start":
                combo_info.info(f"Kombination {upd['index']}/{upd['total']}: "
                                f"**{upd['phase1']} × {upd['phase2']}**")
                bar_total.progress(upd["index"] / max(upd["total"], 1))
                bar_combo.progress(0.0)
            elif upd["type"] == "combination_skip":
                bar_total.progress(upd["index"] / max(upd["total"], 1))
            elif upd["type"] == "progress":
                bar_combo.progress(upd["current"] / max(upd["total"], 1),
                                   text=f"{upd['current']} / {upd['total']}")
            elif upd["type"] == "log":
                icon = "🔴" if upd.get("label") == 1 else "🟢"
                log_entries.insert(0,
                    f"{icon} **ID {upd['id']}** | _{upd.get('text','')[:60]}_\n\n---")
                _render_log(log_entries, log_box)
            elif upd["type"] == "combination_done":
                n_done += 1
                m = upd.get("metrics", {})
                if m:
                    st.success(f"✅ {upd['phase1']} × {upd['phase2']}: "
                               f"Acc {m.get('accuracy',0):.1%} | AUROC {m.get('auroc','n/a')}")
            elif upd["type"] == "all_done":
                st.balloons()
                st.success(f"Fertig! {n_p1} Phase-1-Prompt(s)"
                           + (f", {n_done} Phase-2-Kombinationen" if n_done else "") + " abgeschlossen.")
