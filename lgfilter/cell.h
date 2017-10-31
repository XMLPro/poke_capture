#ifndef CELL_HEADER
#define CELL_HEADER

#include <stdbool.h>

struct cell {
  int x, y;
  bool is_alive;
  int r, g, b;
};

struct game_map {
  int width, height;
  struct cell** cells;
};

void init_game_map(struct game_map*, int w, int h);
void close_game_map(struct game_map*);

#endif
