LIBS = cell.o lifegame.o image.o util.o
FILES = main.o $(LIBS)
OUTPUT = lgfilter
N = 1
I = image.bmp
O = ./output/output.bmp
build: $(FILES)
	cc -O3 -o $(OUTPUT) $(FILES)

share: $(LIBS)
	cc -O3 -shared -fPIC -o lg.so $(LIBS)

run:
	./$(OUTPUT) $(N) -i $(I) -o $(O)

c: $(FILES)
	cc -O3 -o $(OUTPUT) $(FILES)
	./$(OUTPUT) $(N) -i $(I) -o $(O)

.c.o:
	cc -c $<

cell.o: cell.h
lifegame.o: lifegame.h
image.o: image.h
util.o: util.h

clean:
	rm output*.bmp
