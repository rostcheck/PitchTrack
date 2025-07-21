# Makefile for compiling LaTeX documents

# Main document name (without extension)
DOCUMENT = pitch_tracking_research

# LaTeX compiler
LATEX = latexmk

# LaTeX compiler options
LATEX_OPTS = -pdf -interaction=nonstopmode -file-line-error

# Default target
all: $(DOCUMENT).pdf

# Compile the document
$(DOCUMENT).pdf: $(DOCUMENT).tex
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)

# Clean up auxiliary files
clean:
	$(LATEX) -c $(DOCUMENT)

# Clean up all generated files including PDF
distclean:
	$(LATEX) -C $(DOCUMENT)

# View the PDF (macOS specific)
view: $(DOCUMENT).pdf
	open $(DOCUMENT).pdf

.PHONY: all clean distclean view
