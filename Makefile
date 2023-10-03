.PHONY: all install uninstall clean

all: install

install: clean
	pip install -e .

uninstall: clean
	pip uninstall numpynet

clean:
	$(RM) -rf build dist numpynet.egg-info
