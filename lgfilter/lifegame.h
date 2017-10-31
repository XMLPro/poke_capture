#ifndef LIFEGAME_HEADER
#define LIFEGAME_HEADER

#include <stdbool.h>
#include "cell.h"

struct summation_args {
  int** life_map;
  struct game_map* game;
  int x, y;
};

struct cell* get_cell(struct game_map*, int x, int y);
void* summation_x(void*);
void* summation_y(void*);
int** cumlative_sum(struct game_map*);
void close_cumlative_sum(struct game_map*, int** cumsum);
void spawn(struct game_map*, int x, int y);
void kill(struct game_map*, int x, int y);
void next(struct game_map*);

#endif
