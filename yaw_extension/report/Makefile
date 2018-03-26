DEFAULT:
	pdflatex *.tex

all:
	pdflatex *.tex
	bibtex *.aux
	pdflatex *.tex
	pdflatex *.tex

clean:
	rm *.aux
	rm *.log
	rm *.out
	rm *.bbl
	rm *.blg
	rm *.pdf
