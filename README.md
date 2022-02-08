# Thesis article



## Compiling presentation



To compile presentation:

```bash
pandoc -s --slide-level 2 -t beamer thesis_defense_draft.md -o thesis_defense_draft.pdf --pdf-engine lualatex -H preamble_beamer.tex

```

Make sure `lualatex` and `pandoc` are installed, and that the right theme (`Metropolis`) is in the texmf path.