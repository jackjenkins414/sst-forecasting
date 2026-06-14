# Use of Generative AI — Group Declaration

This document records each group member's use of generative AI tools during the project, for inclusion in the report's "Use of Generative AI" section. Each member writes their own statement.

**Shared integrity note:** All model architectures and training algorithms were derived from the cited papers and implemented by the authors. All reported results were produced by the authors' own experiments and verified against the JSON experiment logs; no numerical results in the paper were generated or estimated by AI.

---

## Tools used
- GitHub Copilot
- Claude Code
- NotebookLM (Google)

---

## Per-member statements

### Jack Jenkins (u7333800)
I used Claude (the chat interface) to assist with the coding of my model files and for debugging. Where it contributed to a non-trivial design choice, I flagged this inline in the
source (e.g. the GenAI reference note in `SpaceTimePositionalEncoding` on using reshapes to apply positional encoding along the correct axis). I verified its suggestions myself rather
than taking them on trust: I sanity checked tensor shapes against the expected contract at each boundary (e.g. confirming the patch projection produced `(2, 90, 99, 128)` and that
positional encoding preserved this shape and added no trainable parameters), and fixed bugs it did not catch (e.g. instantiating `nn.Dropout` as a module rather than passing a float). When building `transformer` from the reference source, I also used it to confirm my implementation changes were correct, in particular that the boundary components I modified
for the SST setting were sound while the domain-agnostic blocks (attention, FFN) were carried across faithfully. All architectural decisions and the final implementations are my own.

### Ayush Samuel (u7508601)
I used Github coploit and claude code and for research and building my parts of the LSTM. My work was building the main architecture and framework, AI was used to help in debug specific coding issues. 

### Khaled El-hassan (u6890851)
I used GitHub Copilot and Claude Code as assistants while building and running my parts of the project (the Tubelet Transformer, thehyperparameter-tuning framework, and the long-horizon rollout analysis). My workflow was to design the approach and write the architecture, framework, and initial scripts myself, and to call on AI only when I got stuck on a specific problem. Specifically, I used it to:

- debug errors and look up unfamiliar library APIs;
- adapt scripts to run from PowerShell with command-line `--arguments`, and build job queues so training and tuning runs could execute unattended without manual intervention between runs;
- get targeted help with particular implementation details when a piece of code was not working, having first provided the architectural framework and starter scripts.

I also used it to help draft and tidy portions of the report and to generate quick plotting scripts for results. All architectural decisions, model implementations, experimental
design, and results are my own.

### Isaac Jaensch (u7262835)
I used NotebookLM to extract and categorise key equations from the 'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting' (ProbSparse Informer) paper. This was utilised as a supplementary tool after reading the paper and designing the model framework to assist with code commenting and model verification. NotebookLM was not used as a substitute for the reading nor analysis of the aforementioned paper. 

No Generative AI was used for coding, model design, nor report writing. 
