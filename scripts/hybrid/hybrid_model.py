from .similarity import semantic_sim, structural_sim

def hybrid_score(e1, e2, img1, img2):
    s_sem = semantic_sim(e1, e2)
    s_str = structural_sim(img1, img2)

    # normalize
    s_sem = (s_sem + 1) / 2
    s_str = (s_str + 1) / 2

    # NEW LOGIC (better)
    if s_sem > 0.9:
        return s_sem

    elif s_sem > 0.75:
        return 0.7*s_sem + 0.3*s_str

    else:
        return 0.5*s_sem + 0.5*s_str