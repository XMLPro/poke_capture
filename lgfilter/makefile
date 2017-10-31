FILES = main.o cell.o lifegame.o image.o
OUTPUT = lgfilter
N = 1
I = image.bmp
O = ./output/output.bmp
build: $(FILES)
	cc -O3 -o $(OUTPUT) $(FILES)

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

clean:
	rm output*.bmp
