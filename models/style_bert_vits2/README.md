text (str)
 ├─ normalize
 ├─ text → symbols → text_ids        (T_text)
 ├─ bert(text) → bert_feats          (T_bert)
 ├─ align bert_feats → T_text
 ├─ TextEncoder(
 │     text_ids,
 │     tone_ids,
 │     bert_feats_aligned,
 │     style_vec
 │   ) → prior stats
 ├─ SynthesizerTrn.infer(...)
 └─ wav (44100 Hz)